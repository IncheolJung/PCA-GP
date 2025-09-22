echo $'\n'" === Least RMSE config ==="
echo "  >> number refers to Nth config"
echo "  >> i.e., config N refers to trial 4N-3, 4N-2, 4N-1, and 4N"
echo "  >> This shows best 5 results"

grep "RMSE:" -r ./ | awk -F: '
{
    match($1, /GP_test_*([0-9]+)/, m)
    test_num = m[1]+0
    config = int((test_num-1)/4) + 1   # 4 tests per config
    rmse_sum[config] += $3
}
END {
    for (c in rmse_sum) {
        print "Config", c, "Total RMSE:", rmse_sum[c]
    }
}' | sort -k5,5n | head -n5

echo " ========================="
echo $'\n'" === Fastest convergence config ==="

shopt -s nullglob  # makes *.log expand to nothing if no match

for f in ./GP_test_*/*.log; do
    [ -f "$f" ] || continue

    # extract test number from folder
    test_num=$(echo "$f" | sed -E 's/.*GP_test_0*([0-9]+)\/.*\.log/\1/')
    config=$(( (test_num-1)/4 + 1 ))

    if grep -q "Final iteration:" "$f"; then
        # extract only the first number (before the /)
        iter=$(grep "Final iteration:" "$f" | sed -E 's/.*Final iteration:[[:space:]]*([0-9]+).*/\1/')
        [[ -z "$iter" || "$iter" -eq 0 ]] && iter=20
    else
        iter=20
    fi

    echo "$config $iter"
done | awk '
{
    iter_sum[$1] += $2
}
END {
    for (c in iter_sum) {
        print "Config", c, "Total ITER:", iter_sum[c]
    }
}' | sort -k5,5n | head -n5

echo " =================================="
