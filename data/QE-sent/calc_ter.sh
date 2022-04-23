#! /usr/bin/bash

# original files only
cd ccmt_qe_2019_zh-CN_en

cp -r ../tools/* .
tar xzf tercom-0.7.25.tgz

# tokenize dev
python tokenizeChinese.py dev.src.original dev.src
perl tokenizer.perl -l en -no-escape < dev.tgt.original > dev.tgt
perl tokenizer.perl -l en -no-escape < dev.pe.original > dev.pe
python tokenizeChinese.py dev.src.original dev.src

# prepare for TER calculating
python addLabel.py dev.pe tercom-0.7.25/dev_label.pe
python addLabel.py dev.tgt tercom-0.7.25/dev_label.tgt

# tokenize train
python tokenizeChinese.py train.src.original train.src
perl tokenizer.perl -l en -no-escape < train.tgt.original > train.tgt
perl tokenizer.perl -l en -no-escape < train.pe.original > train.pe

# prepare for TER calculating
python addLabel.py train.pe tercom-0.7.25/train_label.pe
python addLabel.py train.tgt tercom-0.7.25/train_label.tgt

# Calculate TER
cd tercom-0.7.25
java -jar tercom.7.25.jar -r dev_label.pe -h dev_label.tgt -n dev
java -jar tercom.7.25.jar -r train_label.pe -h train_label.tgt -n train
cd ..

# Extract TER
python exactHTER.py tercom-0.7.25/dev.ter dev.hter
python exactHTER.py tercom-0.7.25/train.ter train.hter

# package dataset
tar czf ccmt_qe_2019_zh-CN_en.tar.gz *.original *.src *.tgt *.pe *.hter


# original files only
cd ../ccmt_qe_2019_en_zh-CN

cp -r ../tools/* .
tar xzf tercom-0.7.25.tgz

perl tokenizer.perl -l en -no-escape < dev.src.original > dev.src
python tokenizeChinese.py dev.tgt.original dev.tgt
python tokenizeChinese.py dev.pe.original dev.pe

python addLabel.py dev.pe tercom-0.7.25/dev_label.pe
python addLabel.py dev.tgt tercom-0.7.25/dev_label.tgt

perl tokenizer.perl -l en -no-escape < train.src.original > train.src
python tokenizeChinese.py train.tgt.original train.tgt
python tokenizeChinese.py train.pe.original train.pe

python addLabel.py train.pe tercom-0.7.25/train_label.pe
python addLabel.py train.tgt tercom-0.7.25/train_label.tgt

cd tercom-0.7.25
java -jar tercom.7.25.jar -r dev_label.pe -h dev_label.tgt -n dev
java -jar tercom.7.25.jar -r train_label.pe -h train_label.tgt -n train
cd ..

python exactHTER.py tercom-0.7.25/dev.ter dev.hter
python exactHTER.py tercom-0.7.25/train.ter train.hter

tar czf ccmt_qe_2019_en_zh-CN.tar.gz *.original *.src *.tgt *.pe *.hter
