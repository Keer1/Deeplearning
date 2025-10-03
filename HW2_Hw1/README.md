This repository contains all the necessary files and .py files for the second assignment.

Note:
 To download the trained model. I have added reference link here :
 https://drive.google.com/drive/folders/1SA7ongbIXrrlhqcjL3U3u803QT0Uk3sB?usp=sharing
 
Process to Obtain the BLEU Score:

 Step 1 – Download/Prepare the Model
    The trained model checkpoint (kpatam_seq2seq_best.pth) along with the word–index mappings (w2i_kpatam.pkl, i2w_kpatam.pkl) was used for evaluation.

 Step 2 – Run the Test Script
   The provided shell script hw2_seq2seq.sh internally calls:

    python3 test_seq2seq.py $1 $2

    Here:
       $1 → dataset path (e.g., testing_data/feat)
       $2 → output filename where predictions are stored

 Step 3 – Execute the Script
   For example:
     
     ./hw2_seq2seq.sh testing_data/feat testset_output.txt

This generates a file testset_output.txt containing video IDs and their corresponding predicted captions.

 Step 4 – Evaluate with BLEU
   Run the BLEU evaluation script:
   
    python bleu_eval.py testset_output.txt

Result:
The model achieved an average BLEU score of 0.6458, which is above the baseline of 0.6 specified in the assignment.
 
 
