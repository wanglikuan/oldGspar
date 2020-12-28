python LearnerSimulation.py --model=ResNet18OnCifar10 --workers=15 --epoch=7000 --ratio=0.005 --byzantine=4 --V=0 --method=FABA > 4.out
python LearnerSimulation.py --model=ResNet18OnCifar10 --workers=15 --epoch=7000 --ratio=0.005 --byzantine=5 --V=0 --method=FABA > 5.out
# python LearnerSimulation.py --model=ResNet18OnCifar10 --workers=15 --epoch=7000 --ratio=0.005 --byzantine=3 --V=0 --method=FABA > 3.out

# python LearnerSimulation.py --model=ResNet18OnCifar10 --workers=15 --epoch=1000 --ratio=0.005 --byzantine=1 --V=100 