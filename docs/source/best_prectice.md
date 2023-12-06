This page gives input deck snippets which produce the highest quality fits using TSADAR. It is recomended that when fitting new data a small region of the data is fit with a small number of lineouts. This can be accomplished by setting the `lineouts: start` and `lineouts: end` to be close or increasing `lineouts: skip`. This will alow fast fits that can be used to dial in the starting conditions and the free parameters. This can also be used to check and adjust the fitting ranges. Once the best inital conditions ahve been identified the entire dataset can be fit.

**Best operation for time resolved data:**
```
background:
	type: pixel
	slice: 900
```

**Best operation for spatialy resolved data:**
```
background:
	type: fit
	slice: 900 <or background slice for IAW>
```

**Best operation for lineouts of angular:**
```
background:
	type: fit
	slice: <background shot number>
```

**Best operation for full angular:**
```
background:
	type: fit
	val: <background shot number>
lineouts:
	type: range
	start: 90
  end: 950
```
