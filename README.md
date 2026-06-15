
## Experiments:
- **Baselines**:
	- **Basics**: Random - Greedy - Pere
	- **Papers**: maxvol - bandit 

- **LightGCN embedding** (Implicit Interaction):
	- gowalla - phase 2 - basics 
```
python3 run_epts.py -e phase2 --datasets gowalla --num-proc 16 --run-id gowalla_basics --start-seed 0 --end-seed 1 --update-config
```

	- amazon-books - phase 2 - basics 
```
python3 run_epts.py -e phase2 --datasets amazon_books --num-proc 16 --run-id amazon_books_basics --start-seed 0 --end-seed 1 --update-config
```

	- gowalla - paper 
```
python3 run_epts.py -e dpe --datasets gowalla --num-proc 16 --run-id gowalla_paper --start-seed 0 --end-seed 1 --update-config

python3 run_epts.py -e maxvol --datasets gowalla --num-proc 16 --run-id gowalla_paper --start-seed 0 --end-seed 1 --update-config
```

	- amazon-books - paper
```
python3 run_epts.py -e dpe --datasets amazon_books --num-proc 16 --run-id amazon_books_paper --start-seed 0 --end-seed 1 --update-config

python3 run_epts.py -e maxvol --datasets amazon_books --num-proc 16 --run-id amazon_books_paper --start-seed 0 --end-seed 1 --update-config
```

- **biVAE embedding** (Explicit Interaction): 8000 seconds on amazon books 5 core
	- **amazon-books - phase 2 - basics + dpe + maxvol**
```
python3 run_epts.py -e phase2 --datasets amazon_books_bivae --num-proc 16 --run-id amazon_books_bivae_basics --start-seed 0 --end-seed 1 --update-config

python3 run_epts.py -e dpe --datasets amazon_books_bivae --num-proc 16 --run-id amazon_books_bivae_paper --start-seed 0 --end-seed 1 --update-config

python3 run_epts.py -e maxvol --datasets amazon_books_bivae --num-proc 16 --run-id amazon_books_bivae_paper --start-seed 0 --end-seed 1 --update-config
```

		- amazon-games - phase 2 - basics + dpe + maxvol
```
python3 run_epts.py -e phase2 --datasets amazon_games_bivae --num-proc 16 --run-id amazon_games_bivae_basics --start-seed 0 --end-seed 1 --update-config

python3 run_epts.py -e dpe --datasets amazon_games_bivae --num-proc 16 --run-id amazon_games_bivae_paper --start-seed 0 --end-seed 1 --update-config

python3 run_epts.py -e maxvol --datasets amazon_games_bivae --num-proc 16 --run-id amazon_games_bivae_paper --start-seed 0 --end-seed 1 --update-config
```

- **Ablation Study:** Vary mode (min, max, avg, sum)
	- amazon-books - phase 2 - basics
```
python3 run_epts.py -e vary_mode --datasets amazon_games_bivae --num-proc 16 --run-id amazon_games_bivae_basics --start-seed 0 --end-seed 1 --update-config

python3 run_epts.py -e vary_mode --datasets amazon_books_bivae --num-proc 16 --run-id amazon_games_bivae_basics --start-seed 0 --end-seed 1 --update-config

python3 run_epts.py -e vary_mode --datasets gowalla --num-proc 16 --run-id amazon_games_bivae_basics --start-seed 0 --end-seed 1 --update-config

python3 run_epts.py -e vary_mode --datasets amazon_books --num-proc 16 --run-id amazon_games_bivae_basics --start-seed 0 --end-seed 1 --update-config
```


