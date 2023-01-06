from typing import Protocol
try:
    import tensorflow as tf
except:
    import warnings
    warnings.warn('Tensorflow is not installed. You cannot use the PPA pricer!')
    
#class PPAModel(Protocol):
#    def __init__(self, )

class PPAHedgeModel(tf.keras.Model):
    def __init__(self, model, timegrid, specification, lamda, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.specification = specification # storage constraints
        self.price = tf.Variable([0],trainable=True,dtype ="float32")
        self.timegrid = timegrid
        self.lamda = lamda
        self._prev_q = None

    def __call__(self, x, training=True):
        power_fwd = x[0]
        forecast = x[1]
        return self._compute_pnl(power_fwd, forecast, training) + self.price

    def _compute_pnl(self, power_fwd, forecast, training):
        pnl = 0.0
        self._prev_q = tf.zeros((tf.shape(power_fwd)[0],1), name='prev_q')
        
        for i in range(self.timegrid.shape[0]-1):
            t = self.timegrid[i]
            quantity = self.model([power_fwd, forecast, t], training=training)
            pnl = pnl + (ask_flow * tf.squeeze(ask_prices[:,i])) + (bid_flow * tf.squeeze(bid_prices[:,i])) - ask_flow*self.specification.withdrawalCost - bid_flow*self.specification.injectionCost
            self._current_lvl = self._current_lvl + current_flow
            
        return pnl

class price():

    pass