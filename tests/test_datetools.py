from unittest import main, TestCase
from rivapy.tools.datetools import roll_day, Period, Schedule
from rivapy.tools.enums import RollConvention
from datetime import date
from holidays import DE


class Unit_Tests(TestCase):

    def test_roll_day(self):
        holidays_de = DE()

        # business days are unchanged for all roll conventions
        roll_conventions = [roll_convention.value for roll_convention in RollConvention]
        roll_conventions.pop(2)  # remove 'ModifiedFollowingEOM' as it needs a start date
        for roll_convention in roll_conventions:
            self.assertEqual(roll_day(date(1997, 1, 2), holidays_de, roll_convention), date(1997, 1, 2))

        # test UNADJUSTED
        roll_convention = RollConvention.UNADJUSTED
        self.assertEqual(roll_day(date(1997, 5, 1), holidays_de, roll_convention), date(1997, 5, 1))
        self.assertEqual(roll_day(date(1997, 7, 6), holidays_de, roll_convention), date(1997, 7, 6))

        # test FOLLOWING
        roll_convention = RollConvention.FOLLOWING
        self.assertEqual(roll_day(date(1997, 5, 1), holidays_de, roll_convention), date(1997, 5, 2))
        self.assertEqual(roll_day(date(1997, 7, 5), holidays_de, roll_convention), date(1997, 7, 7))
        self.assertEqual(roll_day(date(1997, 5, 17), holidays_de, roll_convention), date(1997, 5, 20))
        self.assertEqual(roll_day(date(1997, 12, 25), holidays_de, roll_convention), date(1997, 12, 29))
        self.assertEqual(roll_day(date(1997, 3, 28), holidays_de, roll_convention), date(1997, 4, 1))

        # test MODIFIED_FOLLOWING
        roll_convention = RollConvention.MODIFIED_FOLLOWING
        self.assertEqual(roll_day(date(1997, 5, 1), holidays_de, roll_convention), date(1997, 5, 2))
        self.assertEqual(roll_day(date(1997, 12, 25), holidays_de, roll_convention), date(1997, 12, 29))
        self.assertEqual(roll_day(date(1997, 8, 30), holidays_de, roll_convention), date(1997, 8, 29))
        self.assertEqual(roll_day(date(1997, 8, 31), holidays_de, roll_convention), date(1997, 8, 29))
        self.assertEqual(roll_day(date(1997, 3, 28), holidays_de, roll_convention), date(1997, 3, 27))

        # test MODIFIED_FOLLOWING_BIMONTHLY
        roll_convention = RollConvention.MODIFIED_FOLLOWING_BIMONTHLY
        self.assertEqual(roll_day(date(1997, 5, 1), holidays_de, roll_convention), date(1997, 5, 2))
        self.assertEqual(roll_day(date(1997, 12, 25), holidays_de, roll_convention), date(1997, 12, 29))
        self.assertEqual(roll_day(date(1997, 8, 30), holidays_de, roll_convention), date(1997, 8, 29))
        self.assertEqual(roll_day(date(1997, 8, 31), holidays_de, roll_convention), date(1997, 8, 29))
        self.assertEqual(roll_day(date(1997, 3, 28), holidays_de, roll_convention), date(1997, 3, 27))
        self.assertEqual(roll_day(date(1997, 2, 15), holidays_de, roll_convention), date(1997, 2, 14))
        self.assertEqual(roll_day(date(1997, 2, 16), holidays_de, roll_convention), date(1997, 2, 17))

        # test NEAREST
        roll_convention = RollConvention.NEAREST
        self.assertEqual(roll_day(date(1997, 5, 1), holidays_de, roll_convention), date(1997, 5, 2))
        self.assertEqual(roll_day(date(1997, 7, 5), holidays_de, roll_convention), date(1997, 7, 4))
        self.assertEqual(roll_day(date(1997, 7, 6), holidays_de, roll_convention), date(1997, 7, 7))
        self.assertEqual(roll_day(date(1997, 5, 18), holidays_de, roll_convention), date(1997, 5, 20))
        self.assertEqual(roll_day(date(1997, 3, 29), holidays_de, roll_convention), date(1997, 3, 27))
        self.assertEqual(roll_day(date(1997, 3, 30), holidays_de, roll_convention), date(1997, 4, 1))

        # test PRECEDING
        roll_convention = RollConvention.PRECEDING
        self.assertEqual(roll_day(date(1997, 5, 1), holidays_de, roll_convention), date(1997, 4, 30))
        self.assertEqual(roll_day(date(1997, 7, 6), holidays_de, roll_convention), date(1997, 7, 4))
        self.assertEqual(roll_day(date(1997, 5, 19), holidays_de, roll_convention), date(1997, 5, 16))
        self.assertEqual(roll_day(date(1997, 12, 28), holidays_de, roll_convention), date(1997, 12, 24))
        self.assertEqual(roll_day(date(1997, 3, 31), holidays_de, roll_convention), date(1997, 3, 27))

        # test MODIFIED_PRECEDING
        roll_convention = RollConvention.MODIFIED_PRECEDING
        self.assertEqual(roll_day(date(1997, 5, 1), holidays_de, roll_convention), date(1997, 5, 2))
        self.assertEqual(roll_day(date(1997, 7, 5), holidays_de, roll_convention), date(1997, 7, 4))
        self.assertEqual(roll_day(date(1997, 7, 6), holidays_de, roll_convention), date(1997, 7, 4))
        self.assertEqual(roll_day(date(1997, 3, 2), holidays_de, roll_convention), date(1997, 3, 3))

        # test MODIFIED_FOLLOWING_EOM
        roll_convention = RollConvention.MODIFIED_FOLLOWING_EOM
        self.assertEqual(roll_day(date(1997, 3, 28), holidays_de, roll_convention, date(1997, 2, 28)),
                         date(1997, 3, 27))
        self.assertEqual(roll_day(date(1997, 4, 26), holidays_de, roll_convention, date(1997, 3, 26)),
                         date(1997, 4, 28))
        self.assertEqual(roll_day(date(1997, 4, 27), holidays_de, roll_convention, date(1997, 3, 27)),
                         date(1997, 4, 30))
        self.assertEqual(roll_day(date(1997, 6, 29), holidays_de, roll_convention, date(1997, 5, 29)),
                         date(1997, 6, 30))
        self.assertEqual(roll_day(date(1997, 7, 30), holidays_de, roll_convention, date(1997, 6, 30)),
                         date(1997, 7, 31))
        self.assertEqual(roll_day(date(1997, 8, 31), holidays_de, roll_convention, date(1997, 7, 31)),
                         date(1997, 8, 29))
        self.assertEqual(roll_day(date(1997, 9, 28), holidays_de, roll_convention, date(1997, 8, 28)),
                         date(1997, 9, 29))
        self.assertEqual(roll_day(date(1997, 9, 29), holidays_de, roll_convention, date(1997, 8, 29)),
                         date(1997, 9, 30))
        self.assertEqual(roll_day(date(1997, 11, 30), holidays_de, roll_convention, date(1997, 10, 30)),
                         date(1997, 11, 28))
        self.assertEqual(roll_day(date(1997, 12, 28), holidays_de, roll_convention, date(1997, 11, 28)),
                         date(1997, 12, 31))

    def test_schedule_generation(self):
        holidays_de = DE()
        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 3, 0), True, False,
                                  RollConvention.UNADJUSTED, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2020, 11, 21), date(2021, 2, 21), date(2021, 5, 21),
                          date(2021, 8, 21)])
        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 3, 0), True, True,
                                  RollConvention.UNADJUSTED, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2020, 11, 21), date(2021, 2, 21), date(2021, 5, 21),
                          date(2021, 8, 21)])
        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 3, 0), False, False,
                                  RollConvention.UNADJUSTED, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2020, 11, 21), date(2021, 2, 21), date(2021, 5, 21),
                          date(2021, 8, 21)])
        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 3, 0), False, True,
                                  RollConvention.UNADJUSTED, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2020, 11, 21), date(2021, 2, 21), date(2021, 5, 21),
                          date(2021, 8, 21)])

        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 5, 0), True, False,
                                  RollConvention.UNADJUSTED, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2021, 3, 21), date(2021, 8, 21)])
        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 5, 0), True, True,
                                  RollConvention.UNADJUSTED, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2020, 10, 21), date(2021, 3, 21), date(2021, 8, 21)])
        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 5, 0), False, False,
                                  RollConvention.UNADJUSTED, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2021, 1, 21), date(2021, 8, 21)])
        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 5, 0), False, True,
                                  RollConvention.UNADJUSTED, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2021, 1, 21), date(2021, 6, 21), date(2021, 8, 21)])

        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 3, 0), True, False,
                                  RollConvention.MODIFIED_FOLLOWING, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2020, 11, 23), date(2021, 2, 22), date(2021, 5, 21),
                          date(2021, 8, 23)])
        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 3, 0), True, True,
                                  RollConvention.MODIFIED_FOLLOWING, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2020, 11, 23), date(2021, 2, 22), date(2021, 5, 21),
                          date(2021, 8, 23)])
        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 3, 0), False, False,
                                  RollConvention.MODIFIED_FOLLOWING, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2020, 11, 23), date(2021, 2, 22), date(2021, 5, 21),
                          date(2021, 8, 23)])
        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 3, 0), False, True,
                                  RollConvention.MODIFIED_FOLLOWING, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2020, 11, 23), date(2021, 2, 22), date(2021, 5, 21),
                          date(2021, 8, 23)])

        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 5, 0), True, False,
                                  RollConvention.MODIFIED_FOLLOWING, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2021, 3, 22), date(2021, 8, 23)])
        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 5, 0), True, True,
                                  RollConvention.MODIFIED_FOLLOWING, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2020, 10, 21), date(2021, 3, 22), date(2021, 8, 23)])
        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 5, 0), False, False,
                                  RollConvention.MODIFIED_FOLLOWING, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2021, 1, 21), date(2021, 8, 23)])
        self.assertEqual(Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 5, 0), False, True,
                                  RollConvention.MODIFIED_FOLLOWING, holidays_de).generate_dates(False),
                         [date(2020, 8, 21), date(2021, 1, 21), date(2021, 6, 21), date(2021, 8, 23)])


if __name__ == '__main__':
    main()
