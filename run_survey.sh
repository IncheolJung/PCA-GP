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
    for j in {1..5}; do
      for trial in {1..1}; do
        LOG=compute$((i+j+trial)).log
        # grid sampling for variance calc
        python $EXE -a $i -x $j -s 0 -t $term | tee $LOG
        mv $LOG $LOG_DIR
        echo $'\n\n'
      done
    done
  done
done
