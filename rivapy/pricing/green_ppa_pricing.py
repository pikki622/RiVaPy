from typing import Protocol
try:
    import tensorflow as tf
    tf.config.run_functions_eagerly(False)
except:
    import warnings
    warnings.warn('Tensorflow is not installed. You cannot use the PPA pricer!')
    
import numpy as np
import sys
sys.path.append('C:/Users/doeltz/development/RiVaPy/')
import datetime as dt
from rivapy.models import ResidualDemandForwardModel
from rivapy.instruments.ppa_specification import GreenPPASpecification
from rivapy.tools.datetime_grid import DateTimeGrid
from sklearn.preprocessing import StandardScaler
#class PPAModel(Protocol):
#    def __init__(self, )



class PPAHedgeModel(tf.keras.Model):
    def __init__(self, model, timegrid, specification, regularization, strike, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.specification = specification # storage constraints
        self.price = tf.Variable([0], trainable=False, dtype ="float32")
        self.timegrid = timegrid
        self.lamda = regularization
        self._prev_q = None
        self._output_scaler = None
        self.strike = strike
        #self._input_scaler_power = StandardScaler()
        #self._input_scaler_forecast = StandardScaler()

    def __call__(self, x, training=True):
        power_fwd = x[0]
        #power_fwd_scaled = x[1]
        forecast = x[1]
        #forecast_scaled = x[3]
        return self._compute_pnl(power_fwd, forecast, training) #+ self.price
    
    #def get_delta(power_fwd, forecast, t):
    #    power_fwd_scaled = self._input_scaler_power.transform(power_fwd)
    #    forecast_scaled = self._input_scaler_forecast.transform(forecast)

    def _compute_pnl(self, power_fwd, forecast, training):
        pnl = 0.0
        self._prev_q = tf.zeros((tf.shape(power_fwd)[0]), name='prev_q')
        for i in range(self.timegrid.shape[0]-2):
            t = [self.timegrid[i]]*tf.ones((tf.shape(power_fwd)[0],1))/self.timegrid[-1]
            quantity = tf.squeeze(self.model([power_fwd[:,i], forecast[:,i], t], training=training))
            pnl = pnl + tf.math.multiply((self._prev_q-quantity), tf.squeeze(power_fwd[:,i]))
            self._prev_q = quantity
        pnl = pnl + self._prev_q* tf.squeeze(power_fwd[:,-1]) + forecast[:,-1]*(tf.squeeze(power_fwd[:,-1])-self.strike)
            # (-self._prev_q+forecast[:,-1]) * tf.squeeze(power_fwd[:,-1])-forecast[:,-1]*self.strike
        return pnl

    @tf.function
    def custom_loss(self, y_true, y_pred):
        return 0.0*-tf.keras.backend.mean(y_pred) + self.lamda*tf.keras.backend.var(y_true-y_pred)

  
    def train(self, paths_fwd_price, paths_forecasts, lr_schedule, 
        epochs, batch_size, tensorboard_log:str=None, verbose=0):
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) #beta_1=0.9, beta_2=0.999)
        callbacks = []
        if tensorboard_log is not None:
            logdir = tensorboard_log#os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
            callbacks.append(tensorboard_callback)
        self.compile(optimizer=optimizer, loss=self.custom_loss)
        y = np.zeros((paths_fwd_price.shape[0],1))
        return self.fit([paths_fwd_price, paths_forecasts], y, epochs=epochs, 
                            batch_size=batch_size, callbacks=callbacks, verbose=verbose)

def _build_model(depth, nb_neurons):
    power_fwd_price = tf.keras.Input(shape=(1,),name = "power_fwd_price")
    forecast = tf.keras.Input(shape=(1,),name = "forecast")
    t = tf.keras.Input(shape=(1,),name = "t")
    fully_connected_Input = tf.keras.layers.concatenate([power_fwd_price, forecast, t])         
    values_all = tf.keras.layers.Dense(nb_neurons,activation = "selu", 
                    kernel_initializer=tf.keras.initializers.GlorotUniform())(fully_connected_Input)       
    for _ in range(depth):
        values_all = tf.keras.layers.Dense(nb_neurons,activation = "selu", 
                    kernel_initializer=tf.keras.initializers.GlorotUniform())(values_all)            
    value_out = tf.keras.layers.Dense(1, activation="linear",
                    kernel_initializer=tf.keras.initializers.GlorotUniform())(values_all)
    model = tf.keras.Model(inputs=[power_fwd_price, forecast, t], outputs = value_out)
    return model


def price( val_date: dt.datetime,
            green_ppa: GreenPPASpecification,
            power_wind_model: ResidualDemandForwardModel, 
            depth: int, nb_neurons: int, 
            n_sims: int, regularization: float, 
            epochs: int,
            verbose: bool=0,
            tensorboard_logdir: str=None, initial_lr: float = 1e-4, 
            batch_size: int = 100, decay_rate: float=0.7, seed: int = 42):
    #print(locals())
    if green_ppa.technology != power_wind_model.get_technology():
        raise Exception('PPA technology ' + green_ppa.technology + 
                        ' does not equal residual demand technology model ' 
                        + power_wind_model.get_technology())
    if green_ppa.n_deliveries() > 1:
        raise Exception('Pricer for more than one delivery not yet implemented.')
    ppa_schedule = green_ppa.get_schedule()
    if ppa_schedule[-1] <= val_date:
        return None
    tf.random.set_seed(seed)
    model = _build_model(depth, nb_neurons)
    timegrid = DateTimeGrid(start=val_date, end=ppa_schedule[-1], freq='1H')
    rnd = np.random.normal(size=power_wind_model.rnd_shape(n_sims, timegrid.timegrid.shape[0]))
    forecast_points = [i for i in range(len(timegrid.timegrid)) if i%8==0]
    fwd_prices, fwd_residuals = power_wind_model.simulate(timegrid.timegrid, rnd, forecast_points)
    fwd_prices = np.squeeze(fwd_prices.transpose())
    #print(fwd_prices.mean(axis=0))
    forecasts =  np.squeeze((-fwd_residuals+1.0).transpose())
    hedge_model = PPAHedgeModel(model, timegrid.timegrid, None, regularization, strike=green_ppa.fixed_price)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,#1e-3,
            decay_steps=100*fwd_prices.shape[0]/batch_size,
            decay_rate=decay_rate)
    hedge_model.train(fwd_prices, forecasts, lr_schedule, epochs, batch_size, 
                        tensorboard_log=tensorboard_logdir, verbose=verbose)
    return hedge_model

if __name__=='__main__':
    from rivapy.models import WindPowerForecastModel, OrnsteinUhlenbeck
    import numpy as np
    from scipy.special import comb

    def smoothstep(x, x_min=0, x_max=1, N=1):
        x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
        result = 0
        for n in range(0, N + 1):
            result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n
        result *= x ** (N + 1)
        return result


    class SmoothstepSupplyCurve: 
        def __init__(self, s,N):
            self.s = s
            self.N = N
            
        def compute(self, residual, highest_price):
            #wind_production = wind_production#np.maximum(np.minimum(wind_production, 0.99), 0.01)
            #residual = (1.0-wind_production)
            residual = np.power(residual, self.s)
            return smoothstep(residual, N=self.N)*highest_price

    
    days = 2
    timegrid = np.linspace(0.0, days*1.0/365.0, days*24)
    forecast_points = [i for i in range(len(timegrid)) if i%8==0]
    forward_expiries = [timegrid[-1]]
    n_sims = 1_000

    wind_forecast_model = WindPowerForecastModel(speed_of_mean_reversion=0.5, volatility=1.80, 
                                expiries=forward_expiries,
                                forecasts = [0.8, 0.8,0.8,0.8],#*len(forward_expiries)
                                )
    highest_price = OrnsteinUhlenbeck(1.0, 1.0, mean_reversion_level=1.0)
    supply_curve = SmoothstepSupplyCurve(1.0, 0)
    rdm = ResidualDemandForwardModel(wind_forecast_model, highest_price, supply_curve, max_price = 1.0)
    val_date = dt.datetime(2023,1,1)
    spec = GreenPPASpecification(technology = 'Wind_Onshore', 
                            schedule = [val_date + dt.timedelta(days=2)], 
                             fixed_price=0.2)
    model = price(val_date, spec, rdm , depth=3, nb_neurons=64, n_sims = n_sims, regularization= 10.0, 
              epochs = 50, timegrid=timegrid, verbose=1, 
              initial_lr = 1e-2,
              batch_size=200,
              decay_rate=0.6,
              tensorboard_logdir=None
            )
   