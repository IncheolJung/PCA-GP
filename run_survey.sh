EXE=main.py
LOG_DIR=LOGS

# if directory exists, empty it
if [ -d "$LOG_DIR" ]; then
  rm -rf "${LOG_DIR:?}/"*
else
  mkdir -p "$LOG_DIR"
fi

for term in 5; do
  for i in {0..2}; do
    for j in {0..3}; do
      for k in {1..4}; do
        LOG=compute$((i+j+k)).log
        python $EXE -a $i -x $j -t $term | tee $LOG
        mv $LOG $LOG_DIR
      done
    done
  done
done
