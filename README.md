본 코드는 '임베디드 시스템 및 응용' 강좌의 프로젝트 제출코드입니다.

# 실행방법

크게 Database를 생성하는 코드(create_db.py)와 Test dataset에 대해 평가를 진행하는 코드 (main.py)를 실행할 수 있습니다.

[Database 생성]
```
cd visual_localization
export DIR_ROOT=$PWD
export DB_ROOT=/PATH/TO/YOUR/DATASETS
python create_db.py  --dataset Hyundaib1 --checkpoint dirtorch/data/Resnet101-AP-GeM.pt --db db/Hyundai1f_50.npy --gpu 0
```

- dataset : 생성을 원하는 dataset
- checkpoint : deep-retrieval의 backbone checkpoint
- db : db가 생성될 위치

[Test 이미지 평가]
```
cd visual_localization
export DIR_ROOT=$PWD
export DB_ROOT=/PATH/TO/YOUR/DATASETS
python main.py --dataset Hyundaib1 --checkpoint dirtorch/data/Resnet101-AP-GeM.pt --result ./results  --gpu 0
```

- dataset : Test를 진행할 dataset
- checkpoint : deep-retrieval의 backbone checkpoint
- result : 결과물 출력 path