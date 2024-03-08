for fn in {1..133885}
do
    fn=`printf "dsgdb9nsd_%06d" $fn`
    echo $fn
    if [ -e results/$fn/rho_22.npy ]
    then
        echo "skip $fn"
        continue
    fi
    mkdir results/$fn
    cd results/$fn
    python -u ../../scripts/scanner_22.py /path/to/QM9/xyzs/ $fn > scanner.out
    cd ../../
done
