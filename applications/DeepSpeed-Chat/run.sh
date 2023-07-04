#!/bin/bash

# 첫 번째 run.sh 실행
./applications/DeepSpeed-Chat/training/step1_supervised_finetuning/run.sh

# 첫 번째 스크립트가 성공적으로 완료되었을 경우에만 다음 스크립트 실행
if [ $? -eq 0 ]; then
    echo "첫 번째 스크립트가 성공적으로 실행되었습니다."

    # 두 번째 run.sh 실행
    ./applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/run.sh

    # 두 번째 스크립트가 성공적으로 완료되었을 경우에만 다음 스크립트 실행
    if [ $? -eq 0 ]; then
        echo "두 번째 스크립트가 성공적으로 실행되었습니다."

        # 세 번째 run.sh 실행
        ./applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/run.sh

        # 세 번째 스크립트가 성공적으로 완료되었을 경우 메시지 출력
        if [ $? -eq 0 ]; then
            echo "세 번째 스크립트가 성공적으로 실행되었습니다."
        else
            echo "세 번째 스크립트 실행 중 오류가 발생하였습니다."
        fi
    else
        echo "두 번째 스크립트 실행 중 오류가 발생하였습니다."
    fi
else
    echo "첫 번째 스크립트 실행 중 오류가 발생하였습니다."
fi
