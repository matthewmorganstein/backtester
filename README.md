# FlatFire pre-release

*FlatFire* is a blazing phenomenon of financial combustion, where the light of precise trading signals, the heat of strategic backtesting, and the flame of portfolio mastery converge in a radiant, modular API. Built on the "Pointland entry / Sphere exit" strategy, it uses `r_1` and `r_2` indicators to spark trades and spherical logic to exit with precision. Like a star, *FlatFire* illuminates market opportunities, fuels performance, and flares with stunning visualizations, redefining the gravity of financial innovation.

## Features
- **Signal Generation**: Pointland signals using `r_1`/`r_2` > 350 and price breakouts.
- **Backtesting**: Modular `Trade` class with optimized exit logic.
- **Visualization**: Plotly charts for portfolio value and trade events.
- **Scalability**: FastAPI endpoints and async PostgreSQL integration.

## Answers the critical question: "How might this specific strategy have performed in the past?"
Allows for testing and refinement of strategy parameters (square_threshold, distance_threshold) in a risk-free environment.
Builds confidence in a strategy's potential before live deployment, significantly reducing the risk of capital loss from untested ideas.

## Value Delivered:
- Moves beyond subjective assessments to provide concrete, data-backed evidence of strategy performance.
- Provides clear metrics to evaluate whether a strategy meets predefined performance benchmarks.
- Calculates dynamic target and stop levels based on market conditions at the time of the signal (_calculate_target_and_stop using signal_high, signal_low, and distance_threshold).
- Evaluates multiple exit conditions: hitting profit targets, stop-loss levels, or encountering opposing signals (_check_exit_conditions, _determine_exit_outcome).
- Introduces a unique risk metric: time_above_stop, quantifying how long a trade remained favorable relative to its stop-loss level.
- Offers insights into how and why trades exit, revealing patterns in strategy behavior.
- Helps identify potential weaknesses, such as premature exits or vulnerability to specific market conditions.
- The time_above_stop metric provides a nuanced view of a trade's resilience and the proximity to risk, which is often missed in simple win/loss analysis.
- Drastically reduces the time and manual effort required for backtesting compared to spreadsheet-based or manual chart analysis.
- Ensures consistency and repeatability in testing, eliminating human error.
- Frees up valuable analyst/trader time for strategy development and interpretation of results, rather than laborious data processing.

## More
This application serves as a proven proof-of-concept for the upcoming launch of Flatfire. 
