# NEXTAIL Labs code test

This code test is about improving the current solution that is about forecasting the number of products to be ordered in order to anticipate sales. The current stock is also important.  

Current solution is based on the mean value of all previous sales avaiable from a given date.

Two different extra solutions are provided:
- A seasonal solution that takes into cosideration the season or the context of the analysis, which is considered to be a key factor to obtain better results
- A recurrent-based solution that uses a deep model with a gated recurrent unit that learns from the previous trend

To validate the results obtained by all three approaches, the estimated sales are also computed in previous years, so that the given value can be compared with the real value. 

| #time period     | #real sales  | #mean solution  | #mean vs real | #seasonal solution | #seasonal vs real |	#recurrent solution |	#recurrent vs real |
| :--------------: | :----------: | :-------------: | :-----------: | :----------------: | :---------------: | :------------------: | :----------------: |  
| June-August 2017 | 998          | 847.479452    	| -150.520548	  | 961.0     	       | -37.0    	       | 809.478372	          | -188.521628        |
| September 2017   | 198          | 289.648352    	| 91.648352	    | 198.0              | 0.0      	       | 227.159492           | 29.159492          |
| June-August 2018 | 959          | 852.164384    	| -106.835616   | 944.0              | -15.0    	       | 763.736265           | -195.263735        |
| September 2018   | 202          | 291.585366    	| 89.585366     | 200.0              | -2.0     	       | 273.505506           | 71.505506          |
| June-August 2019 | 1054         | 883.397260    	| -170.602740   | 973.666667         | -80.333333	       | 711.898002           | -342.101998        |
| September 2019   | 270          | 297.291139    	| 27.291139     | 223.333333         | -46.666667	       | 275.670955           | 5.670955           |
| June-August 2020 | 1004         | 889.527721    	| -114.472279   | 971.0              | -33.0    	       | 931.660297           | -72.339703         |
| September 2020   | 249          | 296.532148    	| 47.532148     | 229.75             | -19.25            | 297.391365           | 48.391365          |

To run the application, two different options:
- Approach design, implementation and testing: go to https://mybinder.org/ and paste this Github repository name, https://github.com/adiezolivan/other_stuff/, then run 'src/ds_solution_adiezolivan.ipynb'
- Production deployment and execution: run the script 'sales_forecast_main.py' using the following parameters (example given): 
  - order_date = datetime(2020,6,1)
  - lead_time_days = 90
  - days_to_next_order = 30
  - current_stock_level = 400
  - stock_in_transit = 600
  - sales_data = ../sales_data.csv  
  - approach = [mean, seasonal, recurrent]
