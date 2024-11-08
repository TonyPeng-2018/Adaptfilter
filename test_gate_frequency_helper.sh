# i iterates from 0.55 tp 0.95 interval 0.10 and 0.99
for i in $(seq 0.55 0.10 0.95) 0.99
do
    echo "i: $i"
    python test_gate_frequency.py imagenet mobile $i
done
# for i in $(seq 0.55 0.10 0.95) 0.99
# do
#     echo "i: $i"
#     python test_gate_frequency.py cifar-10 mobile $i
# done
# for i in $(seq 0.55 0.10 0.95) 0.99
# do
#     echo "i: $i"
#     python test_gate_frequency.py ccpd mobile $i
# done
for i in $(seq 0.55 0.10 0.95) 0.99
do
    echo "i: $i"
    python test_gate_frequency.py imagenet resnet $i
done
# for i in $(seq 0.55 0.10 0.95) 0.99
# do
#     echo "i: $i"
#     python test_gate_frequency.py cifar-10 resnet $i
# done
# for i in $(seq 0.55 0.10 0.95) 0.99
# do
#     echo "i: $i"
#     python test_gate_frequency.py ccpd resnet $i
# done