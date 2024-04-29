# Data Description  
## Single Question q1 case
- Train 수: 677
- Validaiton 수: 338
- Test 수: 2189
  
## Multiple Questiosn q1, q2, q3 case
- Train 수: 2031
- Validaiton 수: 1014
- Test 수: 6567

Test의 수가 Train의 수보다 훨씬 많다.  
원활한 학습을 위해서 generated_questions_single_q_train.csv나 generated_questions_multiple_q_train.csv가 아닌 generated_questions_single_q_test.csv와 generated_questions_multiple_q_test.csv를 트레이닝 데이터로 쓰고,  
generated_questions_single_q_train.csv나 generated_questions_multiple_q_train.csv를 테스트 데이터로 쓰는게 나아 보인다.

