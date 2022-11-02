rodar:
	python3 impute.py \
        -input output.txt \
        -verbose 2 \
        -hidden_size 512 \
        -rel_mask 0.4 \
        -length 300 \
        -offset $(offset) \
        -max_length 50 \
        -it 70000 \
        -eval_ev 100 \
        -verbose 2