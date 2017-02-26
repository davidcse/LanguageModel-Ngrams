Report:

Q1)
In order to compute Pr_AD(y|x) for cases when (x,y) = 0, we have to use mixing of other methods of smoothing.
This is because if we have try to subtract the discounting factor, which in our case is D=0.5.
If we tried count(x,y)-0.5 when count(x,y)=0, we would have a negative number,
which is unacceptable for a probability p, since p >=0 and p<=1.
By using a mix of other smoothing techniques, such as Katz-Backoff, we are able to account for the unseen events (x,y),
so that we would avoid the above scenario.

Q2)
