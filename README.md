TASK1:

0. Git to get our repo
1. Change the darknet folder's name into darknet_my
2. Download deploy.zip in the folder
3. Unzip the deploy.zip
4. run "python3 gen_voc_label.py"
5. get the darknet: git clone https://github.com/pjreddie/darknet
6. cd darkent
7. replace the Makefile in the darknet by the Makefile under the darknet_my
8. make
9. put the file under darknet_my/cfg/ into darknet/cfg/
10. put the file under darknet_my/data/ into darknet/data/
11. replace the detector.c under darknet/example/ by the detector.c under darkent_my/example/
12. run "wget https://pjreddie.com/media/files/darknet53.conv.74"
13. run "./darknet detector train cfg/my.data cfg/yolov3-my.cfg darknet53.conv.74"
14. You will see the weight in ./backup/ after every 100 iteration
15. run "./darkent detector valid cfg/my.data cfg/yolov3-mytest.cfg ./backup/yolov3-my_xxxx.weights > result.txt"
16. You will see the prediction result for the test data
17. Change the format according and can be submited
(18. if you want to change the resolution for training, also should run python gen_anchor.py to generate the anchor and replace it in the .cfg)

TASK2:

0. uncommand line 493, line 497 in detector.c under example, and command line 494, line 498 in detector.c
1. cd darknet
2. make
3. run "./darkent detector valid cfg/my.data cfg/yolov3-mytest.cfg ./backup/yolov3-my_xxxx.weights > result_new.txt"
4. You will get the result_new.txt for 2d object box position and size
5. cd ..
6. run "python3 task2_code.py"
7. You will get the TASK2.txt for submission 
