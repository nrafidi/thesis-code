#!/bin/sh

# Modified version of Gus's MNE_pipeline.sh
echo "subj dir is $SUBJECTS_DIR"

export SUBJECT=$1 # subject ID, as used in freesurfer subjects directory, e.g. /usr1/meg/structural, or /usr1/apps/freesurfer/subjects
export SPACING=$2 # distance on cortex in mm, between sources to estimate

# 3.3 Cortical surface reconstruction with FreeSurfer . . . . . . . . .                          20

recon-all -subjid $SUBJECT -all | tee -a "$SUBJECT"_surface.log


# 3.4 Setting up the anatomical MR images for MRIlab . . . . . . . .                             20
mne_setup_mri

# 3.5 Setting up the source space . . . . . . . . . . . . . . . . . . . . . . . . .              21
# 3.6 Creating the BEM model meshes . . . . . . . . . . . . . . . . . . . . .                    24
#     Setting up the triangulation files . . . . . . . . . . . . . . . . . . . . . . . .         24
# 3.7 Setting up the boundary-element model . . . . . . . . . . . . . . .                        25

mne_setup_source_space --subject $SUBJECT --spacing $SPACING --overwrite | tee "$SUBJECT"_mne_model.log

# creates the boundary-element model using the Watershed algorithm
# results are file C-head.fif  and directory watershed. after the linking commands below, we'll also have
# inner_skull.surf  outer_skin.surf  outer_skull.surf
mne_watershed_bem --subject $SUBJECT --atlas --overwrite | tee -a "$SUBJECT"_mne_model.log
cd  $SUBJECTS_DIR/$SUBJECT/bem
ln -f -s watershed/"$SUBJECT"_inner_skull_surface "$SUBJECT"-inner_skull.surf
ln -f -s watershed/"$SUBJECT"_outer_skull_surface "$SUBJECT"-outer_skull.surf
ln -f -s watershed/"$SUBJECT"_outer_skin_surface "$SUBJECT"-outer_skin.surf
cd ~

# computes the geometry information for BEM
# results are files such as C-20480-bem.fif, C-20480-bem-sol.fif, inner_skull-20480.pnt, and inner_skull-20480.surf
mne_setup_forward_model --overwrite --homog --subject $SUBJECT --surf --ico 4 | tee -a "$SUBJECT"_mne_model.log

# Create high-density head model for co-registration
mkheadsurf -subjid $SUBJECT -srcvol T1.mgz 
mne_surf2bem --surf $SUBJECTS_DIR/$SUBJECT/surf/lh.seghead --id 4 --check --fif $SUBJECTS_DIR/$SUBJECT/bem/"$SUBJECT"-head-dense.fif
cd  $SUBJECTS_DIR/$SUBJECT/bem
mv -f "$SUBJECT"-head.fif "$SUBJECT"-head-sparse.fif
cp -f "$SUBJECT"-head-dense.fif "$SUBJECT"-head.fif
cd ~

# this is only if we want to be able to display the BEM model in visualizer.
# mne_analyze looks for this specific file name, so we need to be ready for it.
# However, if the BEM file already has a model for the head, it will be loaded
# instead of the one above, so probably it should not be done by default.
#cd  $SUBJECTS_DIR/$SUBJECT/bem
#ln -s $SUBJECT-*-bem.fif $SUBJECT-bem.fif
#cd ~

# after this, the forward calculation in the MNE software computes signals
# detected by each MEG sensor for three orthogonal dipoles at each source space
# location. use raw data for computations, and then average the forawrd
# solutions with mne_average_forward_solutions if necessary

# Gus's own step: create labels for pre-selected regions
mri_annotation2label --subject $SUBJECT --hemi lh --outdir $SUBJECTS_DIR/$SUBJECT/labels/
mri_annotation2label --subject $SUBJECT --hemi rh --outdir $SUBJECTS_DIR/$SUBJECT/labels/
# renaming from ?h.*.label to *-?h.label
cd $SUBJECTS_DIR/$SUBJECT/labels/
for filename in *.label
do
bar=(`echo $filename | tr '.' ' '`)
newname="${bar[1]}"-"${bar[0]}".label
mv -f $filename $newname
done
cd ~
 
