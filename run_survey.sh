EXE=main.py
LOG_DIR=LOGS

# if directory exists, empty it
if [ -d "$LOG_DIR" ]; then
  rm -rf "${LOG_DIR:?}/"*
else
  mkdir -p "$LOG_DIR"
fi

for term in 5; do
  sed -i "328s/.*terms  = .*/    terms  = $term/" $EXE
  for i in {0..0}; do
    sed -i "323s/.*acquisition_function = .*/    acquisition_function = $i/" $EXE
    for j in {2..3}; do
      sed -i "325s/.*Xnormalizer_type = .*/    Xnormalizer_type = $j/" $EXE
      for k in {1..5}; do
        LOG=compute$((i+j+k)).log
        python $EXE | tee $LOG
        mv $LOG $LOG_DIR
      done
    done
  done
done
