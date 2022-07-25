# Highly Dense Sound Event Detection (hd_sed)

Built during the Thesis period for the master of Cognitive Science and Artificial Intelligence (CSAI) in Tilburg University, 
the hd_sed model classsify and temporally-allocate the bird species found in highly dense scenarios. 

The structure of the files and scripts goes as follows: 
- data 
    - dryad
        - annotation
            - Recording_01
            - Recording_02
            - Recording_03
            - Recording_04
        - audio
            - Recording_01
            - Recording_02
            - Recording_03
            - Recording_04
- scripts 
- readme.md

The dryad (Chronister et al., 2021) dataset can be obtained from https://datadryad.org/stash/dataset/doi:10.5061/dryad.d2547d81z

There are 2 options to run the model: 

** Training a new model **          python sed_model.py train <number_of_files> <max_polyphony> <name_of_the_model>
** Testing an existing model **     python sed_model.py test <number_of_files> <max_polyphony> <name_of_the_model>

- The <number_of_files> represents the total number of synthetic audios and annotations built for training and testing 
- The <max_polyphony> represents the maximum number of overlapping sounds found in the dataset (In the paper, 10 was the maximum used)
- The <name_of_the_model> denotes how the model is going to be called after training or which model to load when testing. It's important to add the extension ".h5" on the name.
- The rest of the parameters (architecture, set separation, etc) must be changed in the code (sed_model.py file). It is suggested to change the proportion of training/test in the code when doing a "test" run.  
 