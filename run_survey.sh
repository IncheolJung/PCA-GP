EXE=main.py
LOG_DIR=LOGS
MEMLOG=mem-time.log

# if directory exists, empty it
if [ -d "$LOG_DIR" ]; then
  rm -rf "${LOG_DIR:?}/"*
else
  mkdir -p "$LOG_DIR"
fi

for term in 3; do
  for i in {0..2}; do
    for j in {1..5}; do
      for trial in {1..1}; do
        LOG=compute$((i+j+trial)).log
        # grid sampling for variance calc
        PARMS="-a $i -x $j -s 0 -t $term"
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
