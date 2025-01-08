Codes in here calculate Sherwood for different Peclet and parameter `floor`. (for deeper explanation see Appendix of adjusent publication.)

**Warning:** calculations reuqire folder `output` to be present. This will allow code to save output under `"output/"`.

In order to increase calculation speed GNU PARALELL was used. Run code looked like this:

```console
user:~$cat list_to_calculate.txt | parallel --colsep '\t' python3 calculations.py --peclet {1} --ball_radius {2}
```
