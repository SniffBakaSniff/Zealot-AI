### Scaling / Normalization

Scaling changes the numerical range of data so that a model can train more easily.
One simple method is min-max scaling: 
```
scaled = (value - minimum) / (maximum - minimum)
```
This maps values into a range of approximately 0–1. When using scaled data, the model must be trained and evaluated using the same scale. Predictions can be converted back to the original scale with:
```
original = scaled * (maximum - minimum) + minimum
```
Scaling does not inherently make the model more accurate. It mainly makes the numerical values easier for the optimizer to work with.