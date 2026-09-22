# The Chain Rule
The basic idea of "The Chain Rule is that if A affects B and B affects C then A affects C. We can calculate the overall effect by multiplying the effects together like this:
```
effect of A on C = effect of A on B * effect of B on C
```
This is the chain rule.

For our use case we can calculate 
```
change in output
────────────────
change in weight
``` 
by multiplying 
```
change in hidden        change in output
─────────────────   *   ─────────────────
change in weight        change in hidden
```
The equation for getting hidden and output is.
```
hidden = input * weight
output = hidden * output_weight
```

Some practice problems below:

If were using:
```
input = 2
weight = 3
output_weight = 4
```
1. If we increase weight from 3 to 4, what does hidden become?

        Answer: Hidden becomes 8 because we get our hidden by doing hidden = 2 * 4.

2. If we increase hidden from 6 to 7, what does output become?

        Answer: If we increased the hidden from 6 to 7 the output would become 28 as we calculate the outcome with output = 7 * 4

Using the practice problems answers we can calculate the weights effect on the finale output. So if we get our change in weight and hidden we get these values:
```
change in hidden = 2
change in weight = 1
```
As our hidden went from 6 -> 8 which is a a difference of 2. the same can be done for our change in weight as it went from 3 -> 4 and thats a difference of 1.

Now using the formulas above we can get the weights overall effect by first getting the effect of changing the weight on the hidden value:
```
change in hidden
──────────────── = 2
change in weight
```
and using question 2 we can get these values:
```
hidden: 6 -> 7
output: 24 -> 28
``` 
and with the other equation we can get our change in output:
```
change in output
──────────────── = 4
change in hidden
```
and now that we have both the values we can use the final equation of:
```
A on C = A on B * B on C
```
and that gives us:
```
2 * 4 = 8
```
and finally our final effect of weight on on the output is:
```
change in output
──────────────── = 8
change in weight
```


Basically all of this is so that we can figure out how much the parameter effects the final output and therefore how much said parameter impacts the error/output.



So if we hade these values:

```
input = 5
weight = 2
expected = 45
output_weight = 2
``` 

no using the chain rule we need to figure out these values:
```
hidden = ?
output
error = ?
gradient = ?
```

Lets do this step by step. First we need to get the effect of the weight on the output. so we have `10 = 5 * 2` so our initial weight gives us a hidden of `10`. Now to get the output we do `output = hidden * output_weight` giving us `20 = 10 * 2`, so our initial values gives us:
```
hidden = 10
output = 20
error = -25
gradient = ?
```
Now lets figure out how much the weight effects the output. `15 = 5 * 3` by increasing our weight by `1` the new hidden is now `15` so a `change_in_weight = 1` gives a `change_in_hidden = 15 - 10 = 5` then we do `change_in_hidden / change_in_weight = effect_of_weight_on_hidden` so `5 / 1 = 5`, This means that the weights effect on the hidden is `5` as a change of `1` for the weight changed the hidden by `5`.

Now lets get our error, above we have `20 = 10 * 2` and we get the error by doing `error = output - expected` we get `-25 = 20 - 45` so our error is `-25` now lets increase output by `1` and see what that does to the error. `-24 = 21 - 45` thats a `change_in_output = 1` and a `change_in_error = 1` so the effect of output is `1 / 1 = 1`.

To get the total effect of weight on the output we simply do `weight_effect_on_output = effect_of_weight_on_hidden * effect_of_hidden_on_output` or in this case `10 = 5 * 2`.

To get our gradient we simply do 
`gradient = effect_of_weight_on_hidden * effect_of_hidden_on_output * effect_of_output_on_error` which gives us `10 = 5 * 2 * 1` so our gradient is `10`

