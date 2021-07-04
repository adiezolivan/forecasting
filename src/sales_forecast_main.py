import sys
import logging
import pandas as pd
from dateutil import parser

import sales_forecast_methods


def main() -> int:
    logger = logging.getLogger(__name__)
    logger.info("Main starts")

    try:
        # parameters to be used
        args = sys.argv
        if len(args) == 8:
            order_date = parser.parse(args[1]) #datetime(2020,6,1)
            lead_time_days = int(args[2]) #90
            days_to_next_order = int(args[3]) #30
            current_stock_level = int(args[4]) #400
            stock_in_transit = int(args[5]) #600
            sales_data = pd.read_csv(args[6], parse_dates=['date']) #'../sales_data.csv'
            approach = str(args[7]) #mean, seasonal, recurrent
        else:
            raise Exception("Invalid number of arguments")

        if approach in ['mean', 'seasonal', 'recurrent']:
            if approach == 'mean':
                sol_mean = sales_forecast_methods.calculate_new_order(order_date=order_date,
                                                                      lead_time_days=lead_time_days,
                                                                      days_to_next_order=days_to_next_order,
                                                                      sales_data=sales_data,
                                                                      current_stock_level=current_stock_level,
                                                                      stock_in_transit=stock_in_transit)
                return sol_mean
            elif approach == 'seasonal':
                sol_seasonal = sales_forecast_methods.calculate_new_order_seasonal(order_date=order_date,
                                                                                   lead_time_days=lead_time_days,
                                                                                   days_to_next_order=days_to_next_order,
                                                                                   sales_data=sales_data,
                                                                                   current_stock_level=current_stock_level,
                                                                                   stock_in_transit=stock_in_transit)
                return sol_seasonal
            else:
                sol_recurrent = sales_forecast_methods.calculate_new_order_recurrent(order_date=order_date,
                                                                                     lead_time_days=lead_time_days,
                                                                                     days_to_next_order=days_to_next_order,
                                                                                     sales_data=sales_data,
                                                                                     current_stock_level=current_stock_level,
                                                                                     stock_in_transit=stock_in_transit)
                return sol_recurrent
        else:
            raise Exception("Invalid type of approach")

        return 0
    except Exception as e:
        logger.error("Arguments error: %s", e)
    except Exception as e:
        logger.exception("Unexpected exception: %s", e)

    return -1


if __name__ == "__main__":
    sys.exit(main())
