PY ?= python3

.PHONY: all check figure crosscheck clean
all: crosscheck check figure

crosscheck:            ## mpmath vs scipy Legendre implementations
	$(PY) crosscheck.py

check:                 ## energy conditions and lambda_c  (Sec. 7.3)
	$(PY) energy_conditions.py

figure:                ## horizon-geometry figure and table  (Sec. 5.3)
	$(PY) horizon_figure.py

clean:
	rm -f output/* && touch output/.gitkeep
