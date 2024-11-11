#! bin/bash

for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0
# for i in 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0
# for i in 0.8 0.95
do
    python test_client_speed_new.py mobilenet 1 $i
done

# for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0
for i in 0.0 0.1 0.2 0.3 0.4 0.5
# for i in 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0
# for i in 0.8 0.95
do
    python test_client_speed_new.py mobilenet 2 $i
done

# for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0
# # for i in 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0
# # for i in 0.8 0.95
# do
#     python test_client_speed_new.py resnet 1 $i
# done

# for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0
# # for i in 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0
# # for i in 0.8 0.95
# do
#     python test_client_speed_new.py resnet 2 $i
# done