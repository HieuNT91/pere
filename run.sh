
# run phase 1
python3 run_epts.py --ept phase1 --run-id phase1 --num-proc 32 --update-config --start-seed 0 --end-seed 1


# Run phase 2


# Read results phase 1 -> write csv
python3 read_pkl.py -o results/run_phase1_60/ept_phase1/ -r amazon_books_0_3.pickle

# Read results phase 2 (remember to add --phase2) -> write csv
python3 read_pkl.py --phase2 -o results/run_phase1_60/ept_phase1/ -r amazon_books_0_3.pickle