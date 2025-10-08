EXE=main.py
LOG_DIR=LOGS
MEMLOG=mem-time.log

# if directory exists, empty it
if [ -d "$LOG_DIR" ]; then
  rm -rf "${LOG_DIR:?}/"*
else
  mkdir -p "$LOG_DIR"
fi

for term in 5; do
  for i in {0..0}; do
    for j in {1..2}; do
      for trial in {1..5}; do
        LOG=compute$((i+j+trial)).log
        # grid sampling for variance calc
        PARMS="-a $i -x $j -s 1 -t $term -v -F 0.5e9 1.5e9 101 -i50 -T5e-3"
        # python $EXE $PARMS | tee $LOG
        { /bin/time -v -o $MEMLOG python $EXE $PARMS; } 2>&1 | tee $LOG
        cat $MEMLOG >> $LOG
        mv $LOG $LOG_DIR
        rm $MEMLOG
        echo $'\n\n'
      done
    done
  done
done
