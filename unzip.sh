DIRS="
data/gangwondo
data/jeollado
data/jejudo
data/chungcheongdo
data/gyeongsangdo
"
for DIR in $DIRS; do
    if [ ! -d $DIR ]; then
        echo "make directory train and valid on {$DIR}."
        mkdir $DIR
        mkdir $DIR/train
        mkdir $DIR/valid
    else
        echo "{$DIR} is already exists."
    fi
done

unzip data/"한국어 방언 발화(강원도)"/Training/[라벨]강원도_학습데이터_1.zip -d data/gangwondo/train
unzip data/"한국어 방언 발화(강원도)"/Validation/[라벨]강원도_학습데이터_2.zip -d data/gangwondo/valid
rm -rf data/"한국어 방언 발화(강원도)"/

unzip data/"한국어 방언 발화(전라도)"/Training/[라벨]전라도_학습데이터_1.zip -d data/jeollado/train
unzip data/"한국어 방언 발화(전라도)"/Validation/[라벨]전라도_학습데이터_2.zip -d data/jeollado/valid
rm -rf data/"한국어 방언 발화(전라도)"/

unzip data/"한국어 방언 발화(제주도)"/Training/[라벨]제주도_학습용데이터_1.zip -d data/jejudo/train
unzip data/"한국어 방언 발화(제주도)"/Validation/[라벨]제주도_학습용데이터_3.zip -d data/jejudo/valid
rm -rf data/"한국어 방언 발화(제주도)"/

unzip data/"한국어 방언 발화(충청도)"/Training/[라벨]충청도_학습데이터_1.zip -d data/chungcheongdo/train
unzip data/"한국어 방언 발화(충청도)"/Validation/[라벨]충청도_학습데이터_2.zip -d data/chungcheongdo/valid
rm -rf data/"한국어 방언 발화(충청도)"/

unzip data/"한국어 방언 발화(경상도)"/Training/[라벨]경상도_학습데이터_1.zip -d data/gyeongsangdo/train
unzip data/"한국어 방언 발화(경상도)"/Validation/[라벨]경상도_학습데이터_2.zip -d data/gyeongsangdo/valid
rm -rf data/"한국어 방언 발화(경상도)"/
