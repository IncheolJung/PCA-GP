EXE=main.py
LOG_DIR=LOGS

# if directory exists, empty it
if [ -d "$LOG_DIR" ]; then
  rm -rf "${LOG_DIR:?}/"*
else
  mkdir -p "$LOG_DIR"
fi

for term in 5; do
  for i in {0..0}; do
    for j in {1..3..2}; do
      for trial in {1..5}; do
        LOG=compute$((i+j+trial)).log
        # LHS for sampling in variance calc
        python $EXE -a $i -x $j -s 1 -t $term | tee $LOG
        mv $LOG $LOG_DIR
      done
    done
  done
done
