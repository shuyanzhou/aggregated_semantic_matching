conv_size_arr=(1 2 3 4 5)
pair_type_arr=(1)
for conv_size in ${conv_size_arr[@]}
do
    for pair_type in ${pair_type_arr[@]}
    do
	python3 runner.py --conv_size $conv_size --pair_type $pair_type
    done
done
