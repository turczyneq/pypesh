Codes in here calculate Sherwood for different Peclet na ball_radius. For higher $Pe$ and ball_radius close to 1 it was neccesarry to increase the time of integration for pychastic approach (for deeper explanation see Appendix of adjusent publication.)

**Warning:** calculations expensive in terms of time.

In order to increase calculation speed GNU PARALELL was used. Run code looked like this:

```console
user:~$cat list_to_calculate.txt | parallel --colsep '\t' python3 calculations.py --peclet {1} --ball_radius {2} | tee -a log
```
