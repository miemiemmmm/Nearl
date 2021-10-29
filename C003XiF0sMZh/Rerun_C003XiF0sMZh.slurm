#!/bin/bash -l
#SBATCH --job-name=SMD_XiF0sMZh
#SBATCH --output=/storage006/yzhang/scratch/ACGui/TMP_FOLDERS/XiF0sMZh.out
#SBATCH --error=/storage006/yzhang/scratch/ACGui/TMP_FOLDERS/XiF0sMZh.err
#SBATCH --partition=stromboli
#SBATCH --cpus-per-task=32 --ntasks-per-core=2 --ntasks=3

####Nodelist Here:#SBATCH --nodelist=stromboli[002-006]

# INSERT_VAR_HERE1
# INSERT_VAR_HERE2
ifexp=true
simmode=2
solventmodel=11
prodeng=gmx
ensemble=1
temperature=298.000000
nptsteps=25000
pressure=1
neuiontype=kcl
ionconc=0.150000
initbox=-1
nrsteps=25000000
outfreq=5000
timestep=0.002000
outputformat=netcdf
ifrestart=false
ifreconly=false
nvtsteps=25000
nrcopy=3
rstfrom=false
outputgroup=all
slotsel=6
parmfile=/storage006/yzhang/acgui/TMP_FOLDERS/XiF0sMZh/singleMD/sim_parms.txt
rstfrom=false

# INSERT_VAR_HERE3



# convert the script to string. Remember to double quote when using it. 
# echo "${script_string}"
# cat eiger_restart.slurm | sed -e 's|\$|\\\$|g' -e 's|\\n|\\\\\\n|g' 

# DO IT in the future.  
# TODO: constraints in different simulation mode 
# TODO: use .trr file to gernerate high precision pdb file # TRR outputs 
# TODO: something wrong with the cpumask. Try to correctly use it in the future.  

TMPDIR=/storage006/yzhang/scratch/ACGui/TMP_FOLDERS/XiF0sMZh/
REFDIR=/storage006/yzhang/acgui/TMP_FOLDERS/XiF0sMZh/
SCRIPTDIR=/storage006/yzhang/acgui/scripts/
CAMPDATADIR=/software/campari/data/
CAMPPARM=/software/campari/params/
CAMPEXE=/software/campari/bin/Gnu/campari_mpi
CAMPEXE_NOMPI=/software/campari/bin/Gnu/campari_threads
CAMPMPI_MPI_THREADEXE=/software/campari/bin/Gnu/campari_mpi_threads
CGENFFEXE=/software/SilcsBio/silcsbio.2020.1/cgenff/cgenff
GMXDIR=/software/gromacs_2020.6/ 
export PYTHONPATH=/software/docking/RDKit:/software/ACGui/acgui/scripts
export LD_LIBRARY_PATH=/software/openbabel-openbabel-2-4-1/Gnu/lib:/software/docking/RDKit/lib 
source ${GMXDIR}bin/GMXRC.bash  # Use GMXRC to load Gromacs software 
cp ${SCRIPTDIR}singleMD_ScriptsVars.sh ${REFDIR};
source ${REFDIR}singleMD_ScriptsVars.sh;
cgenffpath=$(tmpstr=""; echo ${CGENFFEXE}| tr '/' '\n' | sed -e /^$/d -e "$ d" | while read i; do tmpstr=${tmpstr}"/"${i}; echo ${tmpstr}; done | tail -1);
export PATH="${cgenffpath}:$PATH";
charmmPath=/storage006/yzhang/tests/test_gromacs/with_lig/charmm36-feb2021.ff/
charmmDirName=$(echo "${charmmPath}" | awk -F '/' '{print $(NF-1)}');
charmmFF=$(echo ${charmmDirName} | sed 's/.ff//g')
tmpfolder=$(echo "${REFDIR}" | awk -F '/' '{print $(NF-1)}'); 


# Check the varialbes needed in different mode. 
if [[ ${ifequil} == 'true' ]] ; then    # equilibration
    if [ -z ${ifequil} ] || [ -z ${eqsctor} ] || [ -z ${equilsteps} ] || [ -z ${neuiontype} ] || [ -z ${ionconc} ] ; then echo "Fatal: No enough argument given" 1>&2; exit 4; fi
elif [[ ${ifimp} == 'true' ]] && [[ ${ifrestart} != "true" ]]; then  # initialize implicit
    if [ -z ${ifimp} ] || [ -z ${simmode} ] || [ -z ${solventmodel} ] || [ -z ${ensemble} ] || [ -z ${ifconst} ] || [ -z ${constres} ] || [ -z ${prodsctor} ] || [ -z ${temperature} ] || [ -z ${inimcsteps} ] || [ -z ${hybmcsteps} ] || [ -z ${hybmdsteps} ] || [ -z ${neuiontype} ] || [ -z ${ionconc} ] || [ -z ${forcefield} ] || [ -z ${nrsteps} ] || [ -z ${outfreq} ] || [ -z ${timestep} ] || [ -z ${outputformat} ] || [ -z ${ifrestart} ] || [ -z ${ifreconly} ] || [ -z ${slotsel} ] ; then echo "Fatal: No enough argument given in implicit solvent simulation" 1>&2; exit 4; fi
elif [[ ${ifexp} == "true" ]] && [[ ${ifrestart} != "true" ]]; then  # initialize explicit 
    if [ -z ${ifexp} ] || [ -z ${simmode} ] || [ -z ${solventmodel} ] || [ -z ${prodeng} ] || [ -z ${ensemble} ] || [ -z ${temperature} ] || [ -z ${nvtsteps} ] || [ -z ${nptsteps} ] || [ -z ${pressure} ] || [ -z ${neuiontype} ] || [ -z ${ionconc} ] || [ -z ${initbox} ] || [ -z ${nrsteps} ] || [ -z ${outfreq} ] || [ -z ${timestep} ] || [ -z ${outputformat} ] || [ -z ${ifrestart} ] || [ -z ${ifreconly} ] || [ -z ${slotsel} ]; then echo "Fatal: No enough argument given in explicit solvent simulation" 1>&2; exit 4 ; fi
elif [[ ${ifimp} == 'true' ]] && [[ ${ifrestart} == "true" ]]; then   # restart implicit 
    echo restart the simulation with implicit solvent simulation
elif [[ ${ifexp} == 'true' ]] && [[ ${ifrestart} == "true" ]]; then   # restart explicit
    if [ -z ${pressure} ] || [ -z ${temperature} ] || [ -z ${nrsteps} ] || [ -z ${outfreq} ] || [ -z ${timestep} ] || [ -z ${outputformat} ] || [ -z ${ifrestart} ] || [ -z ${nrcopy} ] || [ -z ${rstfrom} ] || [ -z ${outputgroup} ]; then echo "Fatal: No enough argument given for restarting an explicit solvent simulation" 1>&2; exit 4 ; fi  
fi 

# Setup for constraints and FRZfile generation -> only Available for INT constraints now.
ifconst=false
if [[ ${ifconst} == 'true' ]]; then 
    echo "Preparing the constrain file";
    rm -f ${REFDIR}singleMD/single_consres.frz;  
    echo "A" >> ${REFDIR}singleMD/single_consres.frz; 
    removespace=$(echo ${constres}|sed 's/ //g'); 
    respairs=$(echo  ${removespace}| tr "," "\n"); 
    echo "The constrianted residues: ${constres}";
    for i in ${respairs}; do 
        if [[ ${i} =~ - ]]; then 
            tmpstr=$(echo $i|tr "-" "\n");
            foreresnum=$(echo ${tmpstr}|awk -n '{print $1}');
            laterresnum=$(echo ${tmpstr}|awk -n '{print $2}');
            for j in $(seq ${foreresnum} ${laterresnum}); do 
                awk -v id=${j} '{if ($6 == id) print($2, " INT")}' ${REFDIR}singleMD/rec.pdb >> ${REFDIR}singleMD/single_consres.frz 
            done
        else
            awk -v id=${i} '{if ($6 == id) print($2, " INT")}' ${REFDIR}singleMD/rec.pdb >> ${REFDIR}singleMD/single_consres.frz
        fi
    done
fi

# Examine the restart session. 
rstfrom=${rstfrom:-"false"} 
if [ ${#rstfrom} -eq 12 ] && [ -d ${REFDIR}singleMD/${rstfrom} ]; then echo "A restart simulation from ${rstfrom} will be executed";
elif [ ${#rstfrom} -eq 0 ] || [ ${rstfrom} == "false" ] || [ ${rstfrom} == "no" ]; then echo "A new simulation will be executed";
else echo "Fatal: Unexpected length of the restart session or restart folder not found" 1>&2; exit 66;
fi 

rm -f ${REFDIR}singleMD/target_wat_ion.seq  ${REFDIR}singleMD/rec.pdb  ${REFDIR}singleMD/trajLigand.mol2;
ifequil=${ifequil:-"false"};  # Set to false when ifequil not defined
slotsel=${slotsel:-0};        # Set to DEFAULT when slotsel not defined
if [[ ${ifequil} == 'true' ]]; then  slotsel=${sim_eqslotsel};  fi   # NOTE: only use the slotsel variable. 

# Determine the structure input files from different slots
if [ ${slotsel} -eq 6 ] ; then  
    echo "Using the previous equilibrated structure as MD input";
    pdbfile=${REFDIR}singleMD/sloteq/rec.pdb;
    ligfile=${REFDIR}singleMD/sloteq/trajLigand.mol2; 
    fileCheck ${pdbfile} "Deposited structure" 64;
    if [[ ${ifreconly} != "true" ]] && [ -f ${ligfile} ]; then  fileCheck ${ligfile} "Equilibration structure" 64;  
    else  ifreconly="true"; ligfile="";
    fi
    seqfile=${REFDIR}singleMD/sloteq/target_wat_ion.seq;

elif [ ${slotsel} -ge 1 ] && [ ${slotsel} -le 5 ] ; then 
    echo "Using the previously deposited structure in slot ${slotsel} as MD input";
    pdbfile=${REFDIR}singleMD/slot${slotsel}/rec.pdb;
    ligfile=${REFDIR}singleMD/slot${slotsel}/trajLigand.mol2;
    fileCheck ${pdbfile} "Deposited structure" 64;
    if [[ ${ifreconly} != "true" ]] && [ -f ${ligfile} ]; then  fileCheck ${ligfile} "Deposited structure" 64; 
    else  ifreconly="true"; ligfile="";
    fi 
    seqfile=${REFDIR}singleMD/slot${slotsel}/target_wat_ion.seq; 

elif [ ${slotsel} -eq 0 ] ; then 
    echo "Using default structure as input";
    mkdir -p ${REFDIR}singleMD/slot0/;
    cp ${REFDIR}tmp_Sampling_target.pdb ${REFDIR}singleMD/slot0/rec.pdb;
    cp ${REFDIR}tmp_Sampling_target.mol2 ${REFDIR}singleMD/slot0/trajLigand.mol2; 
    cp ${REFDIR}tmp_Sampling_target.seq  ${REFDIR}singleMD/slot0/target_wat_ion.seq; 

    pdbfile=${REFDIR}singleMD/slot0/rec.pdb;
    ligfile=${REFDIR}singleMD/slot0/trajLigand.mol2; 
    fileCheck ${pdbfile} "Default structure" 64; 
    if [[ ${ifreconly} != "true" ]] && [ -f ${ligfile} ]; then  fileCheck ${ligfile} "Deposited structure" 64;
    else  ifreconly="true"; ligfile="";
    fi
    seqfile=${REFDIR}singleMD/slot0/target_wat_ion.seq;
fi

# If not restart: Re-parameter the ligand file confirmed by the previous step
# If restart: Extract the previous structure is standard input structure. 
ifreconly=${ifreconly:-"false"};
sed -i "s|ifreconly=[a-z]*$|ifreconly=${ifreconly}%|g" ${parmfile};
if [[ ${ifrestart} != "true" ]] && [[ ${ifreconly} != "true" ]]; then
    rm -f ${REFDIR}singleMD/abs.mol2 ${REFDIR}singleMD/python.log ${REFDIR}singleMD/python2.log;
    echo "Re-parameterizing mol2 files";
    cgenfflog=$(cat ${ligfile} | ${CGENFFEXE} -m /dev/null -i mol2| egrep "ATOM|LONEPAIR" | awk -v kk=0 '{if ($1 == "ATOM") {kk=kk+1; print kk,$3,$4,$2,$6} else if ($1 == "LONEPAIR") {print $3,$4,$5,$7}}' | sed -e 's|!||g');
    python2.7 ${SCRIPTDIR}decon_wrapper.py  ${SCRIPTDIR}ions_neutrals_corrected.csv ${SCRIPTDIR}correctives_values.csv "${cgenfflog}" ${ligfile} ${REFDIR}singleMD/abs.mol2 >& ${REFDIR}singleMD/python.log;
    if [[ $(wc ${REFDIR}singleMD/abs.mol2 | awk '{print $1}') -gt 10 ]]; then
        mv  ${REFDIR}singleMD/abs.mol2  ${REFDIR}singleMD/trajLigand.mol2; 
    elif [[ -n "/storage006/jremy/corina" ]]; then
        /storage006/jremy/corina -i t="mol2" ${ligfile} -d newtypes -d no3d -o t="mol2" > ${REFDIR}singleMD/tmp_ligand_corina.mol2
        python2.7 ${SCRIPTDIR}decon_wrapper.py  ${SCRIPTDIR}ions_neutrals_corrected.csv ${SCRIPTDIR}correctives_values.csv "${cgenfflog}" ${REFDIR}singleMD/tmp_ligand_corina.mol2  ${REFDIR}singleMD/abs.mol2 >& ${REFDIR}singleMD/python2.log;
        if [[ $(wc ${REFDIR}singleMD/abs.mol2 | awk '{print $1}') -gt 10 ]]; then  mv ${REFDIR}singleMD/abs.mol2  ${REFDIR}singleMD/trajLigand.mol2;
        else  sed -e "s|1  UNL1|0 LIG |" ${ligfile} > ${REFDIR}singleMD/trajLigand.mol2; 
        fi
    else
        sed -e "s|1  UNL1|0 LIG |" ${ligfile} > ${REFDIR}singleMD/trajLigand.mol2;
    fi
    ligfile=${REFDIR}singleMD/trajLigand.mol2;
    fileCheck ${ligfile} "Ligand re-parameterization" 8; 

elif [[ ${ifrestart} == "true" ]]; then  # The file preparation in restart simulation;
    # In restart simulation, should not use the ${rstfrom}_PDB.pdb because partial output may damage the original topology, use the checkpoint.pdb instead.
    # Restart only supports GMX restart, hence, no need for ligand re-parameterization
    rm -f  ${REFDIR}singleMD/checkpoint.*;
    tar --ignore-failed-read -xvf ${REFDIR}singleMD/${rstfrom}/${rstfrom}_RSTPKG.tar -C ${REFDIR}singleMD/ checkpoint.pdb checkpoint.mol2 checkpoint.seq; 
    if [[ ${ifreconly} != "true" ]] ; then fileCheck ${REFDIR}singleMD/checkpoint.mol2 "Restart package " 63; fi
    fileCheck ${REFDIR}singleMD/checkpoint.pdb  "Restart package " 63;
    fileCheck ${REFDIR}singleMD/checkpoint.seq  "Restart package " 63;

    mv ${REFDIR}singleMD/checkpoint.pdb ${REFDIR}singleMD/rec.pdb;
    mv ${REFDIR}singleMD/checkpoint.seq ${REFDIR}singleMD/target_wat_ion.seq;
    pdbfile=${REFDIR}singleMD/rec.pdb;
    seqfile=${REFDIR}singleMD/target_wat_ion.seq;
    if [ -f checkpoint.mol2 ]; then  mv ${REFDIR}singleMD/checkpoint.mol2 ${REFDIR}singleMD/trajLigand.mol2;  ligfile=${REFDIR}singleMD/trajLigand.mol2; fi 
fi

mkdir -p ${TMPDIR};
cd ${TMPDIR};
fileCheck ${pdbfile} "PDB input file" 67;
fileCheck ${seqfile} "Sequence file" 67;
echo "seq: ${seqfile}"; echo "pdb: ${pdbfile}"; echo "lig: ${ligfile}";
# Set the file according to the starting structure slot they select. 

# Unify the name ligand to: Trag_Lig and Check the max length of the atom name 
# Automatically remove the _[0-9] to normal atom name 
cat ${ligfile}
checklp=$(egrep "LP|Du" ${ligfile});
if [ ${#checklp} -ne 0 ]; then 
    python3 -c  """${cleanmol2}""" ${ligfile} ${TMPDIR}tmp.mol2; 
    if [ -f ${TMPDIR}tmp.mol2 ]; then mv ${TMPDIR}tmp.mol2 ${ligfile}; fi 
fi 
if [[ ${ifreconly} != "true" ]] && [ ${#ligfile} -gt 0 ]; then 
    fileCheck ${ligfile} "Mol2 ligand file" 67; 
    Ligname=$(grep -A1 '@<TRIPOS>MOLECULE' ${ligfile} | tail -1 | awk '{print $1}') && sed -i "s/${Ligname}/Traj_Lig/"  ${ligfile};
    awk '{if ($2 ~ "_"){ gsub(/_[0-9]*\t*/, " ", $2); print } else { print }}' ${ligfile} > ${TMPDIR}tmp.mol2 && mv ${TMPDIR}tmp.mol2 ${ligfile}; 
    awk '{if ($2 ~ "_"){ gsub(/_[0-9]*\t*/, " ", $2); print } else { print }}' ${REFDIR}tmp_Sampling_target.mol2  > ${TMPDIR}tmp.mol2 && mv ${TMPDIR}tmp.mol2 ${REFDIR}tmp_Sampling_target.mol2;
fi

if [ -s ${REFDIR}autoPatch_PatchKeys.txt ]; then  srun -v --hint=compute_bound --cpus-per-task=${SLURM_CPUS_PER_TASK} -n ${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 --ntasks-per-core=${SLURM_NTASKS_PER_CORE} --mpi=pmix_v2 rsync --ignore-missing-args -au ${REFDIR}autoPatch_* ${TMPDIR}/ ; fi

lastNr=$(ls -d  ${REFDIR}singleMD/[CTRL][0-9][0-9][0-9]${tmpfolder} | awk -F / '{print $(NF)}' | awk '{print substr($0,2,3)}' | sort -n | tail -1 ) && lastNr=$(echo ${lastNr:-0}); 

echo "Last Number is ${lastNr}" 1>&2
#+++++++++++++++++>>>>>>>>>>>>>>>>> After this line, Try not to directly use the absolute path; Use structure file as variable instead. 

# Equilibration simulation
if [[ ${ifequil} == 'true' ]] && [[ ${ifrestart} != 'true' ]]; then
    echo "Starting the structure equilibration at $(date +%b.%d.%y-%H:%M);";
    rm -rf ${REFDIR}singleMD/sloteq;
    pdbcenter=$(egrep "ATOM|HETATM" ${pdbfile} | awk -v x=0 -v y=0 -v z=0 -v kk=0 '{if (substr($0,1,4) == "ATOM") {x=x+1.0*substr($0,31,8); y=y+1.0*substr($0,39,8); z=z+1.0*substr($0,47,8); kk=kk+1}; if (kk > 0) {print x/kk,y/kk,z/kk}}' | tail -n1);
    length=$(grep "ATOM|HETATM" ${pdbfile} | awk -v cen="${pdbcenter}" 'BEGIN{maxd=50; split(cen,cxyz); cx=1.0*cxyz[1]; cy=1.0*cxyz[2]; cz=1.0*cxyz[3]} {if ((substr($0,1,4) == "ATOM") || (substr($0,1,6) == "HETATM")) {x=1.0*substr($0,31,8); y=1.0*substr($0,39,8); z=1.0*substr($0,47,8); dis=(x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz); if (sqrt(dis) > maxd) {maxd = sqrt(dis);}}} END{print maxd}');
    halfstep=$(echo "${equilsteps}/2" | bc);
    systemvol=$(echo "${length}" | awk '{print 4.18879*$1*$1*$1}');
    electrolyteSystem ${neuiontype} ${ionconc} ${systemvol} ${REFDIR} ${seqfile} ${pdbfile} ${ligfile};

    echo -e ${imp_sim_key} | tr '%' '\n' | sed -e "s|scriptdir|${SCRIPTDIR}|g" -e "s|campadatadir|${CAMPDATADIR}|g" -e "s|campaparamsdir|${CAMPPARM}|g" -e "s|tmpfolder|${REFDIR}|g" -e "s|THESEQFILE|${seqfile}|g" -e "s|THEPDBFILE|${pdbfile}|g" -e "s|THEMOL2FILE|${ligfile}|g" -e "s|THEBASENAME|SMDEQ|g" \
    -e "s|THESIZE|${length}|g" -e "s|THEORIGIN|${pdbcenter}|g" -e "s|THESHAPE|2|g" -e "s|THEBOUND|4|g" -e "s|THENRSTEP|${equilsteps}|g" -e "s|THEEQUIL|${equilsteps}|g" -e "s|THEDYNAMICS|5|g" -e "s|THE_TEMPERATURE|300|g" -e "s|THE_TIMESTEP|0.002|g" \
    -e "s|THEMCFIRST|${halfstep}|g" -e "s|THE_MC_CYCLE|50|g" -e "s|THE_MD_CYCLE|200|g" -e "s|THE_ENSEMBLE|1|g" -e "s|THETRAJFORMAT|5|g" -e "s|THEOUTFREQ|${equilsteps}|g" -e "s|THEIFLIGAND|1|g" -e "s|PARMFILE|abs4.2_charmm36.prm|g" -e "s|TMD_NRRELAX .*|TMD_NRRELAX 200|g" -e "s|RIGIDRDFREQ .*|RIGIDRDFREQ 0.0|g" \
    -e "s|PIVOTRDFREQ .*|PIVOTRDFREQ 0.0|g" -e "s|OTHERRDFREQ .*|OTHERRDFREQ 0.0|g" -e "s|NUCRDFREQ .*|NUCRDFREQ 0.0|g" > ${REFDIR}singleMD/SMD_equil.key

    if [[ ${ifreconly} == "true" ]]; then
        sed -i -e "s|MOL2SCREEN.*|MOL2SCREEN 0|g" ${REFDIR}singleMD/SMD_equil.key;
        sed -i /LIG_N_C/d ${seqfile};
    fi

    if (( $(echo "${eqsctor} > 0"|bc) )); then 
        torfile=${REFDIR}singleMD/EQTorFile.txt; PrepareTorfile ${torfile} 20 ${pdbfile}; 
        sed -i -e "s|TORFILE .*|TORFILE ${torfile}|g" -e "s|SC_TOR .*|SC_TOR ${eqsctor}|g" ${REFDIR}singleMD/SMD_equil.key ; 
    else 
        sed -i -e /SC_TOR/d -e /TORFILE/d ${REFDIR}singleMD/SMD_equil.key; 
    fi

    if [[ ${ifconst} == 'true' ]]; then 
        echo "FMCSC_FRZFILE ${REFDIR}singleMD/single_consres.frz" >> ${REFDIR}singleMD/SMD_equil.key; echo "FMCSC_FRZREPORT 1" >> ${REFDIR}singleMD/SMD_equil.key ;
    fi
    addPatches ${TMPDIR} ${REFDIR}singleMD/SMD_equil.key;

    srun -v --hint=compute_bound  -n 1 --cpus-per-task=${SLURM_CPUS_PER_TASK}  --ntasks-per-core=${SLURM_NTASKS_PER_CORE} --mpi=pmix_v2 ${CAMPMPI_MPI_THREADEXE} -k ${REFDIR}singleMD/SMD_equil.key >& ${TMPDIR}SMDSIM_equil.log

    varcheck1=$(grep "Fatal. The input sequence for a small molecule screen should end with a single residue molecule not natively supported by CAMPARI)." ${TMPDIR}N_000.log) 
    if [ ${#varcheck1} -gt 0 ]; then echo "Unsupport residue found when generating initial structure by CAMPARI"; exit 50; fi

    rankNr=$(printf "%03d" ${SLURM_PROCID}); 
    if [[ ${ifreconly} == "true" ]]; then 
        fileCheck ${TMPDIR}N_${rankNr}_SMDEQ_END.pdb "PDB structure from equilibration" 85; 
        mkdir -p ${REFDIR}singleMD/sloteq;
        cp ${TMPDIR}N_${rankNr}_SMDEQ_END.pdb ${REFDIR}singleMD/sloteq/rec.pdb; 
    else 
        fileCheck ${TMPDIR}N_${rankNr}_Traj_Lig_END.pdb "PDB structure from equilibration" 85;
        fileCheck ${TMPDIR}N_${rankNr}_SMDEQ.mol2 "Mol2 output from equilibration" 85;
        mkdir -p ${REFDIR}singleMD/sloteq;
        cp ${TMPDIR}N_${rankNr}_Traj_Lig_END.pdb ${REFDIR}singleMD/sloteq/rec.pdb;
        if [ -f ${TMPDIR}N_${rankNr}_SMDEQ.mol2 ]; then cp ${TMPDIR}N_${rankNr}_SMDEQ.mol2 ${REFDIR}singleMD/sloteq/trajLigand.mol2; fi  # output the equilibrated structure. 
    fi 
    cp ${seqfile} ${REFDIR}singleMD/sloteq/target_wat_ion.seq; 
    rm -f ${TMPDIR}N_[0-9][0-9][0-9]*
    exit 1;

elif [[ ${ifimp} == 'true' ]] && [[ ${ifrestart} != 'true' ]] ; then   # Implicit solvent simulation
    echo "Initiating an implicit solvent simulation at $(date +%b.%d.%y-%H:%M)";
    if [[ ${forcefield} == "ff1" ]]; then forcefield=abs4.2_charmm36.prm; 
    elif [[ ${forcefield} == "ff2" ]]; then forcefield=abs4.2_opls.prm; 
    else forcefield=abs4.2_charmm36.prm; 
    fi 
    if [[ ${outputformat} == "netcdf" ]]; then outputformat=5; trajext=nc;
    elif  [[ ${outputformat} == "xtc" ]]; then outputformat=4; trajext=xtc;
    elif  [[ ${outputformat} == "dcd" ]]; then outputformat=3; trajext=dcd;
    elif  [[ ${outputformat} == "pdb" ]]; then outputformat=2; trajext=pdb;
    fi

    if [ ${slotsel} -ge 1 ] && [ ${slotsel} -le 6 ]; then  rstfrom="false";  fi 
    sed -i "s|rstfrom.*$|rstfrom=${rstfrom}|g" ${parmfile};

    pdbcenter=$(egrep "ATOM|HETATM" ${pdbfile} | awk -v x=0 -v y=0 -v z=0 -v kk=0 '{if (substr($0,1,4) == "ATOM") {x=x+1.0*substr($0,31,8); y=y+1.0*substr($0,39,8); z=z+1.0*substr($0,47,8); kk=kk+1}; if (kk > 0) {print x/kk,y/kk,z/kk}}' | tail -n1);
    systemlength=$(grep "ATOM|HETATM" ${pdbfile} | awk -v cen="${pdbcenter}" 'BEGIN{maxd=50; split(cen,cxyz); cx=1.0*cxyz[1]; cy=1.0*cxyz[2]; cz=1.0*cxyz[3]} {if ((substr($0,1,4) == "ATOM") || (substr($0,1,6) == "HETATM")) {x=1.0*substr($0,31,8); y=1.0*substr($0,39,8); z=1.0*substr($0,47,8); dis=(x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz); if (sqrt(dis) > maxd) {maxd = sqrt(dis);}}} END{print maxd}');
    systemvol=$(echo "${systemlength}" | awk '{print 4.18879*$1*$1*$1}');   

    # Normalize the pdb file intermediate MD structure (preprocess to startdard long) / CTDock structure / SEED structure (use the original) 
    if [[ ${rstfrom} != "true" ]] && [ ${slotsel} -ge 1 ] && [ ${slotsel} -le 6 ]; then 
        pdbNormalize ${pdbfile} ${TMPDIR} ${REFDIR} ${REFDIR}singleMD/tmp.pdb && mv ${REFDIR}singleMD/tmp.pdb ${REFDIR}singleMD/rec.pdb && pdbfile=${REFDIR}singleMD/rec.pdb; 
        electrolyteSystem ${neuiontype} ${ionconc} ${systemvol} ${REFDIR} ${seqfile} ${pdbfile} ${ligfile}; 
    elif [[ ${rstfrom} != "true" ]] && [ ${slotsel} -eq 0 ]; then 
        electrolyteSystem ${neuiontype} ${ionconc} ${systemvol} ${REFDIR} ${seqfile} ${pdbfile} ${ligfile};
    elif [[ ${rstfrom} == "true" ]]; then 
        pdbNormalize ${pdbfile} ${TMPDIR} ${REFDIR} ${REFDIR}singleMD/tmp.pdb && mv ${REFDIR}singleMD/tmp.pdb ${REFDIR}singleMD/rec.pdb && pdbfile=${REFDIR}singleMD/rec.pdb; 
        electrolyteSystem ${neuiontype} ${ionconc} ${systemvol} ${REFDIR} ${seqfile} ${pdbfile} ${ligfile}; 
    fi 

    echo "systemlength=${systemlength}" >> ${parmfile};  # when initialize the system, give systemlength to the parameter file for further restart.
    echo -e ${imp_sim_key} | tr '%' '\n' | sed -e "s|scriptdir|${SCRIPTDIR}|g" -e "s|campadatadir|${CAMPDATADIR}|g" -e "s|campaparamsdir|${CAMPPARM}|g" -e "s|tmpfolder|${REFDIR}|g" \
    -e "s|THESEQFILE|${seqfile}|g" -e "s|THEPDBFILE|${pdbfile}|g" -e "s|THEMOL2FILE|${ligfile}|g" \
    -e "s|THEBASENAME|SMDIMP|g" -e "s|THESIZE|${systemlength}|g" -e "s|THEORIGIN|${pdbcenter}|g" -e "s|THESHAPE|2|g" -e "s|THEBOUND|4|g" \
    -e "s|THENRSTEP|${nrsteps}|g" -e "s|THEEQUIL|0|g" -e "s|THEDYNAMICS|${simmode}|g" -e "s|THE_TEMPERATURE|${temperature}|g" -e "s|THE_TIMESTEP|${timestep}|g" \
    -e "s|THEMCFIRST|${inimcsteps}|g" -e "s|THE_MC_CYCLE|${hybmcsteps}|g" -e "s|THE_MD_CYCLE|${hybmdsteps}|g" \
    -e "s|THE_ENSEMBLE|${ensemble}|g" -e "s|THETRAJFORMAT|${outputformat}|g" -e "s|THEOUTFREQ|${outfreq}|g" \
    -e "s|THEIFLIGAND|1|g" -e "s|PARMFILE|${forcefield}|g" > ${REFDIR}singleMD/IMP_prod.key

    if [[ ${ifreconly} == "true" ]]; then 
        sed -i -e "s|MOL2SCREEN.*|MOL2SCREEN 0|g" ${REFDIR}singleMD/IMP_prod.key; 
        sed -i /LIG_N_C/d ${seqfile};
    fi 

    if (( $(echo "${prodsctor} > 0"|bc) )); then torfile=${REFDIR}singleMD/TorFile.txt; PrepareTorfile ${torfile} 20 ${pdbfile}; sed -i -e "s|TORFILE .*|TORFILE ${torfile}|g" -e "s|SC_TOR .*|SC_TOR ${prodsctor}|g" ${REFDIR}singleMD/IMP_prod.key; fi
    addPatches ${TMPDIR} ${REFDIR}singleMD/IMP_prod.key;

    # prepare script file and parameters need by this script. 
    echo "Implicit simulation script starts at $(date +%b.%d.%y-%H:%M)";

    cat ${SCRIPTDIR}smd_impprod.sh > ${REFDIR}singleMD/IMP_Production.sh; 
    echo "TMPDIR=${TMPDIR}%REFDIR=${REFDIR}%SCRIPTDIR=${SCRIPTDIR}%CAMPDATADIR=${CAMPDATADIR}%CAMPPARM=${CAMPPARM}%ifreconly=\"${ifreconly}\"%trajext=${trajext}%seqfile=${seqfile}%CAMPEXE_NOMPI=${CAMPEXE_NOMPI}%CAMPMPI_MPI_THREADEXE=${CAMPMPI_MPI_THREADEXE}%nrsteps=${nrsteps}%timestep=${timestep}%outputformat=${outputformat}%forcefield=${forcefield}%neuiontype=${neuiontype}%outputgroup=${outputgroup}%systemlength=${systemlength}%keyfile=${REFDIR}singleMD/IMP_prod.key%lastNr=${lastNr}%parmfile=${parmfile}%slotsel=${slotsel}%outfreq=${outfreq}" | tr "%" "\n" > ${TMPDIR}tmp.txt; 
    insertInd=$(cat -n ${REFDIR}singleMD/IMP_Production.sh | grep INSERT_VARIABLE_HERE | awk '{print $1}');
    sed -i "${insertInd}r  ${TMPDIR}tmp.txt" ${REFDIR}singleMD/IMP_Production.sh; 
    chmod +x ${REFDIR}singleMD/IMP_Production.sh;
    srun --wait=7200 -v  -n ${nrcopy} --cpus-per-task=${SLURM_CPUS_PER_TASK}  --ntasks-per-core=${SLURM_NTASKS_PER_CORE} --mpi=pmix_v2 ${REFDIR}singleMD/IMP_Production.sh > ${TMPDIR}IMP_productions.log 2>&1;
    checklast=$(ls -d  ${REFDIR}singleMD/[CTRL][0-9][0-9][0-9]${tmpfolder} | awk -F / '{print $(NF)}' | awk '{print substr($0,2,3)}' | sort -n | tail -1 );
    if [ ${lastNr} -eq ${checklast} ]; then  echo "Fatal: No new Trajectory is detected."; exit 62; fi
    last_id=$(ls -d  ${REFDIR}singleMD/[CTRL][0-9][0-9][0-9]${tmpfolder} | awk -F / '{print $(NF)}' | sort -nk1.2,1.4 | tail -1 );
    echo "In the past simulation, the latest generated trajectory ID is ${last_id}";
    echo "Implicit simulation ends at $(date +%b.%d.%y-%H:%M)";

    check1=$(egrep "Fatal:|exit code|The designated file.*is not found or too small" ${TMPDIR}IMP_productions.log);
    Exit_code=$(egrep "Fatal:|exit code|The designated file.*is not found or too small" ${TMPDIR}IMP_productions.log | sed "s|.*exit code ||g" | awk '{print $1}');
    if [ ${#check1} -ne 0 ]; then  Exit_code=${Exit_code:-54}; exit ${Exit_code}; fi

    check2=$(egrep "Simulation Success" ${TMPDIR}IMP_productions.log | head -1 );
    if [ ${#check2} -ne 0 ]; then  echo "Simulation successed"; exit 1;
    else  echo "No explicit exit code – Fail in later stage"; exit 68; 
    fi

elif [[ ${ifexp} == 'true' ]] && [[ ${ifrestart} != 'true' ]]; then
    # Here initialize an explicit solvent simulation
    if [ -z ${ifexp} ] || [ -z ${simmode} ] || [ -z ${solventmodel} ] || [ -z ${prodeng} ] || [ -z ${ensemble} ] || [ -z ${temperature} ] || [ -z ${nptsteps} ] || [ -z ${pressure} ] || [ -z ${neuiontype} ] || [ -z ${ionconc} ] || [ -z ${initbox} ] || [ -z ${nrsteps} ] || [ -z ${outfreq} ] || [ -z ${timestep} ] || [ -z ${outputformat} ] || [ -z ${ifrestart} ] || [ -z ${ifreconly} ] ; then echo "No enough argument given" 1>&2; exit 4 ; fi
    rm -f ${TMPDIR}em.* ${TMPDIR}npt.* ${TMPDIR}\#*\#;  # Files from Gromacs mdrun and automatic backup
    rm -f ${TMPDIR}*CRASH.* ${TMPDIR}*RELEAVE*;         # previous CAMPARI temporary file
    rm -f ${TMPDIR}*chain* ${TMPDIR}topol* ${TMPDIR}mdl* ${TMPDIR}*.ndx;
    rm -f ${TMPDIR}rec_solved*.pdb;                     # Temporary initial structure files from CAMPARI;  
    rm -f ${TMPDIR}GMX_Executions.log;
    cp -r ${charmmPath} ${TMPDIR}; python3 -c """${patchFF}"""  ${TMPDIR}${charmmDirName}; 
    cp ${SCRIPTDIR}cgenff_charmm2gmx_py3_nx1.py ${REFDIR}singleMD/;


    forcefield=charmm36.prm; 
    if   [[ ${forcefield} == "ff1" ]]; then forcefield=charmm36.prm; 
    elif [[ ${forcefield} == "ff2" ]]; then forcefield=amber94.prm; 
    elif [[ ${forcefield} == "ff3" ]]; then forcefield=amber99.prm; 
    elif [[ ${forcefield} == "ff4" ]]; then forcefield=GROM53a6.prm; 
    else forcefield=charmm36.prm; 
    fi 

    if [[ ${outputformat} == "netcdf" ]]; then outputformat=5; trajext=nc; 
    elif  [[ ${outputformat} == "xtc" ]]; then outputformat=4; trajext=xtc; 
    elif  [[ ${outputformat} == "dcd" ]]; then outputformat=3; trajext=dcd; 
    elif  [[ ${outputformat} == "pdb" ]]; then outputformat=2; trajext=pdb; 
    fi

    if [ ${solventmodel} == 11 ]; then watCode='T3P'; wat_itp='tip3p'; 
    elif [ ${solventmodel} == 12 ]; then watCode='SPC'; wat_itp='spc'; 
    else echo "No valid water model is given in explicit solvent simulation" 1>&2; exit 6; 
    fi


    if [ ${neuiontype} = "nacl" ]; then catCode='NA+'; aniCode='CL-'; catGMX='NA '; aniGMX='CL ';
    elif [ ${neuiontype} = "kcl" ]; then catCode='K+'; aniCode='CL-'; catGMX='K '; aniGMX='CL ';
    else echo "No valid ion type, it should be a problem in variable transfer" 1>&2; exit 6;
    fi 

    spliter "Step 1: Generate the initial structure by CAMPARI $(date +%b.%d.%y-%H:%M)";
    # Normalize the pdb file intermediate MD structure (preprocess to startdard long) / CTDock structure / SEED structure (use the original) 
    # TODO: What if user change the receptor and we work on the new receptor  
    # In this logic, using the tmp_sampling target as reference, this could work. 
    pdbNormalize ${pdbfile} ${TMPDIR} ${REFDIR} ${REFDIR}singleMD/tmp.pdb && mv ${REFDIR}singleMD/tmp.pdb ${REFDIR}singleMD/rec.pdb && pdbfile=${REFDIR}singleMD/rec.pdb; 
    cp ${REFDIR}tmp_Sampling_target.seq ${REFDIR}singleMD/target_wat_ion.seq; seqfile=${REFDIR}singleMD/target_wat_ion.seq;
    if [ ${wat_itp} != 'tip3p' ] && [ ${wat_itp} == 'spc' ]; then  sed -i 's/T3P/SPC/g' ${pdbfile}; sed -i 's/T3P/SPC/g' ${seqfile}; fi 

    echo "Determining the solvent box and ions"
    grep ATOM ${pdbfile} | awk '{print substr($0,31,24)}' > ${TMPDIR}tmp.txt;
    coor_output=$(python3 -c """${getDimensions}""" ${TMPDIR}tmp.txt) && echo "getDimensions: Getting the dimension information about the protein structure";
    box_center=$(echo ${coor_output}| tr '%' ' '| awk '{print $3}'| tr '_' ' '); 
    if [ ${initbox} -gt 0 ]; then
        new_box=$(echo ${initbox} ${initbox} ${initbox});
        new_origin=$(python3 -c """import sys; lcenter=sys.argv[1:4]; lbox=sys.argv[4:7]; c=[float(lcenter[i])-(float(lbox[i])/2) for i in range(3)]; print(' '.join([str(i) for i in c])) """ ${box_center} ${new_box});
    else
        new_box=$(echo ${coor_output}| tr '%' ' '| awk '{print $2}'| tr '_' ' ');
        new_origin=$(echo ${coor_output}| tr '%' ' '| awk '{print $5}'| tr '_' ' ');
    fi

    if [ ${#new_box} -eq 0 ] || [ ${#new_origin} -eq 0 ]; then "Fatal: failed to get the dimension of protein system." 1>&2; exit 41; fi 
    echo "New box size: ${new_box}, New box origin: ${new_origin}; Box center: ${box_center};"

    rec_Info=$(recWeight ${pdbfile});
    rec_Weight=$(echo ${rec_Info} | tr '%' ' ' | awk '{print $1}');
    recVolume=$(echo ${rec_Info} | tr '%' ' ' | awk '{print $2}');
    echo "Receptor weight and volume: ${rec_Info}";
    numWatIon=$(python3 -c """${getWatIonNum}"""  $(echo ${new_box}| tr ' ' '%')  $( echo ${ionconc} *1000 | bc ) ${recVolume}) && echo 'getWatIonNum: Calculating the number of ions/water needed for the box';
    waternum=$(echo ${numWatIon} | tr '%' ' '| awk '{print $1}');
    ionnum=$(echo ${numWatIon} | tr '%' ' ' | awk '{print $2}');
    recNetCharge=$(recCharge ${pdbfile} ${seqfile} ${REFDIR} ${ifreconly})
    recNetCharge=$(python3 -c """import sys; print(round(float(sys.argv[1]), 4))""" ${recNetCharge});
    absCharge=$(python3 -c """import sys; print(abs(round(float(sys.argv[1]))))""" ${recNetCharge});
    echo "receptor net charge ${recNetCharge}" 1>&2
    if   (( $(echo "${recNetCharge} > 0" |  bc -l) )); then   catnum=${ionnum} ; aninum=$(echo ${ionnum} + ${absCharge} | bc);
    elif (( $(echo "${recNetCharge} < 0" |  bc -l) )); then  catnum=$(echo ${ionnum} + ${absCharge} | bc) ; aninum=${ionnum};
    elif (( $(echo "${recNetCharge} == 0" | bc -l) )); then  catnum=${ionnum} ; aninum=${ionnum};
    fi
    if [ ${#waternum} -eq 0 ] || [ ${#recNetCharge} -eq 0 ] || [ ${#catnum} -eq 0 ] || [ ${#aninum} -eq 0 ] ; then "Fatal: failed to get the solvent information of simulation system." 1>&2; exit 9;  fi
    echo "Weight of receptor: ${rec_Weight} Dalton; Volume of the protein: ${recVolume}; Netcharge of receptor: ${recNetCharge}";
    echo "Number of water molecules: ${waternum}, cation: ${catnum}, anion: ${aninum}";

    echo "Preparing the sequence file for CAMPARI input";
    WatStr1=$(getSolStr ${waternum} ${watCode}); cationStr=$(getSolStr ${catnum} ${catCode}); anionStr=$(getSolStr ${aninum} ${aniCode});
    if [[ ${ifreconly} == "true" ]]; then i_index=$(cat -n ${seqfile} | grep END | awk '{print $1-1}'); 
    elif [[ ${ifreconly} != "true" ]]; then i_index=$(cat -n ${seqfile} | grep LIG_N_C | awk '{print $1-1}');
    fi 

    insertStr1=${WatStr1}${cationStr}${anionStr};
    echo ${insertStr1} > ${TMPDIR}tmp.txt;
    sed -i "${i_index}r ${TMPDIR}tmp.txt" ${seqfile} && sed -i "s/%/\n/g" ${seqfile} && sed -i /^$/d ${seqfile};

    if [ ${#WatStr1} -eq 0 ] || [ ${#insertStr1} -eq 0 ]; then "Fatal: failed to get the input string for struction of ion atoms." 1>&2; exit 9; fi

    # Prepare the ligand files. and remove duplicates inside. 
    if [[ ${ifreconly} != "true" ]]; then
        echo "GROMACS: Processing ligand $(date +%b.%d.%y-%H:%M)";
        cp ${ligfile} ${TMPDIR}pure_lig.mol2 && sed -i 2d ${TMPDIR}pure_lig.mol2 && sed -i '2i MDL' ${TMPDIR}pure_lig.mol2;

        checkliglp=$(egrep "LP|Du" ${TMPDIR}pure_lig.mol2);
        if [ ${#checkliglp} -ne 0 ]; then
            python3 -c  """${cleanmol2}""" ${TMPDIR}pure_lig.mol2  ${TMPDIR}tmp.mol2;
            if [ -f ${TMPDIR}tmp.mol2 ]; then mv ${TMPDIR}tmp.mol2  ${TMPDIR}pure_lig.mol2; fi
        fi

        cat ${TMPDIR}pure_lig.mol2 | ${CGENFFEXE} -m /dev/null -i mol2 > ${TMPDIR}MDL.str;

        checkmdlstr=$(egrep "LP|Du" ${TMPDIR}MDL.str); 
        if [ ${#checkmdlstr} -ne 0 ]; then sed -i -e /LP/d -e /Du/d ${TMPDIR}MDL.str; fi 

        python3 ${REFDIR}singleMD/cgenff_charmm2gmx_py3_nx1.py MDL  ${TMPDIR}pure_lig.mol2  ${TMPDIR}MDL.str  ${TMPDIR}${charmmDirName};
        gmx -quiet -nobackup editconf -f ${TMPDIR}mdl_ini.pdb -o ${TMPDIR}mdl.gro >> ${TMPDIR}GMX_Executions.log 2>&1;
        grep MDL ${TMPDIR}mdl.gro > ${TMPDIR}smdlig.str;

        echo -e "0 & ! a H* \n q" | gmx make_ndx -f ${TMPDIR}mdl.gro -o ${TMPDIR}index_mdl.ndx  >> ${TMPDIR}GMX_Executions.log 2>&1 ;
        echo "3" | gmx genrestr -f ${TMPDIR}mdl.gro -n ${TMPDIR}index_mdl.ndx -o ${TMPDIR}posre_mdl.itp -fc 1000 1000 1000  >> ${TMPDIR}GMX_Executions.log 2>&1;

        # Normally speaking, if success to generate the ligand, there should not be duplicate in the ligand. 
        duplicate=$(cat ${ligfile} | sed -ne '/<TRIPOS>ATOM/,/<TRIPOS>BOND/ p' | sed '/<TRIPOS>/d'| awk '{print $2}'| uniq -c | sort -n -k1 | tail -1 | awk '{print $1}');
        if [ ${duplicate} -gt 1 ]; then python3 -c """${mol2NameByCgenff}""" ${TMPDIR}pure_lig.mol2  ${TMPDIR}MDL.str ${TMPDIR}mdl_ini.pdb  ${TMPDIR}tmp.mol2 && sed -i '2s/.*/Traj_Lig/' ${TMPDIR}tmp.mol2 && mv ${TMPDIR}tmp.mol2 ${ligfile}; echo "There is atom name duplicate in the input mol2 file, updating the original file."; fi
        fileCheck ${TMPDIR}smdlig.str "Ligand string file" 32;
        fileCheck ${TMPDIR}posre_mdl.itp "Ligand positional restraint" 32;
        fileCheck ${TMPDIR}mdl.prm "Ligand Gromacs parameter" 32;
        fileCheck ${TMPDIR}mdl.itp "Ligand itp file" 32;
    fi 

    # Step 1: Generate the initial structure by CAMPARI.
    python3 -c """${prep_frz}""" ${pdbfile} ${TMPDIR}rec_restraints.frz && echo 'python->prep_frz: prepared the constraint file for CAMPARI';
    fileCheck ${TMPDIR}rec_restraints.frz  "FRZ file for equilibration"  12;
    echo -e ${getCampInp_key} | tr '%' '\n' | sed -e "s|campadatadir|${CAMPDATADIR}|g" -e "s|campaparamsdir|${CAMPPARM}|g" -e "s|PARMFILE|${forcefield}|g" \
        -e "s|THESEQFILE|${seqfile}|g" -e "s|THEPDBFILE|${pdbfile}|g" -e "s|THEBOXSIZE|${new_box}|g" -e "s|THEORIGIN|${new_origin}|g" -e "s|THENRSTEP|1|g" -e "s|FMCSC_PDB_W_CONV.*|FMCSC_PDB_W_CONV 2|g" \
        -e "s|THEBASENAME|SYS_ASSEMBLE|g" -e "s|THESHAPE|1|g" -e "s|THEDYNMODE|1|g" -e "s|THEEQUIL|0|g" -e "s|THEBOUND|1|g" -e "s|CAMPMOL2FILE|${ligfile}|g" -e "s|THEFRZFILE|${TMPDIR}rec_restraints.frz|g" > ${REFDIR}singleMD/EXP_sysassemble.key;
    addPatches ${TMPDIR} ${REFDIR}singleMD/EXP_sysassemble.key;
    echo -e "# Hamiltonian and cutoffs%FMCSC_CUTOFFMODE 4 # topology-assisted cutoffs%FMCSC_MCCUTMODE 2 # safer %FMCSC_NBCUTOFF 12%FMCSC_ELCUTOFF 12%FMCSC_NBL_UP 5%FMCSC_LREL_MD 4%FMCSC_LREL_MC 3%FMCSC_N2LOOP 0%FMCSC_UAMODEL 0%FMCSC_INTERMODEL 1%FMCSC_ELECMODEL 3%FMCSC_MODE_14 1%FMCSC_FUDGE_EL_14 1.0%FMCSC_FUDGE_ST_14 1.0%FMCSC_EPSRULE 2%FMCSC_SIGRULE 1%FMCSC_MOL2BONDMODE 3%FMCSC_PLANAR_TOLS 15. 30.%" | tr '%' '\n' >> ${REFDIR}singleMD/EXP_sysassemble.key;
    if [[ ${ifreconly} == "true" ]]; then  sed -i /LIG_N_C/d ${seqfile}; sed -i -e "s|MOL2SCREEN.*|MOL2SCREEN 0|g" ${REFDIR}singleMD/EXP_sysassemble.key; fi

    srun -v --hint=compute_bound  -n ${SLURM_NTASKS} --cpus-per-task=${SLURM_CPUS_PER_TASK} --ntasks-per-core=${SLURM_NTASKS_PER_CORE} --mpi=pmix_v2 ${CAMPEXE_NOMPI} -k ${REFDIR}singleMD/EXP_sysassemble.key >& ${TMPDIR}System_Assemble.log && rm -f ${TMPDIR}SYS_ASSEMBLE*START*; 
    echo "Structural initialization completed at $(date +%b.%d.%y-%H:%M)";

    if [[ ${ifreconly} != "true" ]]; then 
        fileCheck ${TMPDIR}Traj_Lig_END.pdb "System Assemble Output" 21;
        grep LIG ${TMPDIR}Traj_Lig_END.pdb > ${TMPDIR}LIG_CAMPOrder.pdb;  
        mv ${TMPDIR}Traj_Lig_END.pdb ${TMPDIR}rec_assembled.pdb && sed -i /LIG/d ${TMPDIR}rec_assembled.pdb;
        fileCheck ${TMPDIR}LIG_CAMPOrder.pdb "CAMPARI Preferred Ligand PDB" 22; 
    elif [[ ${ifreconly} == "true" ]]; then 
        fileCheck ${TMPDIR}SYS_ASSEMBLE_END.pdb "System Assemble Output" 21;
        mv ${TMPDIR}SYS_ASSEMBLE_END.pdb ${TMPDIR}rec_assembled.pdb; 
    fi
    
    echo "HERE TEST" 1>&2
    # Step 1.2: Solve clashes between protein/ions and water
    # Extract water/protein/ion coordinate -> identify clashes -> remove clashes respectively
    egrep "OW.*T3P|OW.*SPC|OW.*T4P|OW.*T4E|OW.*T5P" ${TMPDIR}rec_assembled.pdb | awk '{print substr($0,7,5), substr($0,23,4), substr($0,31,8), substr($0,39,8), substr($0,47,8)}'| awk '{print $1,$2,$3,$4,$5}' > ${TMPDIR}wat_str.cor;
    egrep "OW.*T3P|OW.*SPC|OW.*T4P|OW.*T4E|OW.*T5P" ${pdbfile} | awk '{print substr($0,7,5), substr($0,23,4), substr($0,31,8), substr($0,39,8), substr($0,47,8)}' | while read i; do re=$(echo ${i} | tr " " "%"|sed 's|%|\.\*|g'); sed -i "/${re}/d" ${TMPDIR}wat_str.cor; done 
    grep ATOM ${TMPDIR}rec_assembled.pdb | awk '{if (substr($0,14,1)!="H"){ print substr($0,7,5), substr($0,23,4), substr($0,31,8), substr($0,39,8), substr($0,47,8) }}' | awk '{print $1,$2,$3,$4,$5}' > ${TMPDIR}pro_str.cor;
    grep -e "${catCode}" -e "${aniCode}" ${TMPDIR}rec_assembled.pdb | awk '{print substr($0,7,5), substr($0,23,4), substr($0,31,8), substr($0,39,8), substr($0,47,8)}' | awk '{print $1,$2,$3,$4,$5}' > ${TMPDIR}ion_str.cor;
    python3 -c """${find_clash_script}""" ${TMPDIR}wat_str.cor ${TMPDIR}pro_str.cor PROTEIN 3.0 ${TMPDIR}proclashes.cor; 
    python3 -c """${find_clash_script}""" ${TMPDIR}wat_str.cor ${TMPDIR}ion_str.cor ION 2.5 ${TMPDIR}ionclashes.cor;  
    cat ${TMPDIR}proclashes.cor ${TMPDIR}ionclashes.cor | sort -n -k2 | uniq > ${TMPDIR}clashes.cor;
    python3 -c """${rmClashWat}""" ${TMPDIR}clashes.cor ${TMPDIR}rec_assembled.pdb ${TMPDIR}rec_solved1.pdb && echo 'python->find_clash_script/rmClashWat: solving calshes (protein/ions)'
    if ! [ -f ${TMPDIR}ionclashes.cor ] || ! [ -f ${TMPDIR}proclashes.cor ]; then echo "Python script for clash solving did not work properly"; exit 11; fi
    # The coordinate file might be empty (Zero clash), however, the file should be generated : cannot use filecheck
    fileCheck ${TMPDIR}clashes.cor "Clashes between solvent ions and ion/protein" 11;
    fileCheck ${TMPDIR}rec_solved1.pdb "PDB structure that removes clashes " 9;
    rm -f ${TMPDIR}wat_str.cor ${TMRPDIR}pro_str.cor ${TMPDIR}ion_str.cor ${TMPDIR}proclashes.cor ${TMPDIR}ionclashes.cor;

    # Step 1.3: NVT equilibration with CAMPARI software 
    # Continue with previous rec_solved1.pdb -> rebuild the sequence file -> NVT simulation
    cp ${REFDIR}tmp_Sampling_target.seq  ${REFDIR}singleMD/target_wat_ion.seq;
    if [ ${wat_itp} != 'tip3p' ] && [ ${wat_itp} == 'spc' ]; then sed -i 's/T3P/SPC/g' ${pdbfile}; sed -i 's/T3P/SPC/g' ${seqfile}; fi

    nWaterTotal=$(grep OW ${TMRPDIR}rec_solved1.pdb | wc -l);  nWaterRec=$(grep OW  ${pdbfile} | wc -l);
    solWater=$(echo ${nWaterTotal}-${nWaterRec} | bc);  WatStr2=$(getSolStr ${solWater} ${watCode}); 

    insertStr2=${WatStr2}${cationStr}${anionStr};
    echo ${insertStr2} > ${TMPDIR}tmp.txt;

    echo "The system has ${nWaterTotal} water molecules in total; User defined water: ${nWaterRec}; Added water: ${solWater}";
    if [[ ${ifreconly} == "true" ]]; then i_index=$(cat -n ${seqfile} | grep END | awk '{print $1-1}'); 
    elif [[ ${ifreconly} != "true" ]]; then i_index=$(cat -n ${seqfile} | grep LIG_N_C | awk '{print $1-1}');
    fi 
    sed -i "${i_index}r ${TMPDIR}tmp.txt" ${seqfile} && sed -i "s/%/\n/g" ${seqfile} && sed -i /^$/d ${seqfile} && sed -i /LIG_N_C/d ${seqfile}; 

    echo -e ${releave_water_key} | tr '%' '\n' | sed -e "s|campadatadir|${CAMPDATADIR}|g" -e "s|campaparamsdir|${CAMPPARM}|g" -e "s|PARMFILE|${forcefield}|g" -e "s|THESEQFILE|${seqfile}|g" -e "s|THEPDBFILE|${TMPDIR}rec_solved1.pdb|g" -e "s|BOX_SIZE|${new_box}|g" -e "s|ORIGIN_VECTOR|${new_origin}|g" -e "s|THENRSTEP|${nvtsteps}|g" -e "s|THEEQUIL|$(echo ${nvtsteps}+1|bc)|g" -e "s|THEBASENAME|RELEAVE|g" -e "s|THEFRZFILE|${TMPDIR}rec_restraints.frz|g"  -e "s|FMCSC_PDB_W_CONV.*|FMCSC_PDB_W_CONV 2|g"  > ${REFDIR}singleMD/EXP_releavewat.key; 
    addPatches ${TMPDIR} ${REFDIR}singleMD/EXP_releavewat.key; 
    srun -v  -n ${SLURM_NTASKS} --cpus-per-task=${SLURM_CPUS_PER_TASK}  --ntasks-per-core=${SLURM_NTASKS_PER_CORE} --mpi=pmix_v2 ${CAMPMPI_MPI_THREADEXE} -k ${REFDIR}singleMD/EXP_releavewat.key >& ${TMPDIR}WAT_RELEAVE.log 
    echo "Equilibration of water box with CAMPARI NVT simulation (with receptor constrained) finished at $(date +%b.%d.%y-%H:%M)";
    oneEndPdb=$(ls ${TMPDIR}N_[0-9][0-9][0-9]_RELEAVE_END.pdb | tail -1);
    if [ ${#oneEndPdb} -eq 0 ]; then echo "Fatal: The end structure from CAMPARI NVT simulation is missing.";exit 24; fi 
    mv ${oneEndPdb} ${TMPDIR}WatEquil.pdb && fileCheck ${TMPDIR}WatEquil.pdb "NVT equilibrated PDB structure" 24; 

    mv ${TMPDIR}N_000.log ${TMPDIR}NVT_log.txt;
    rm -f ${TMPDIR}N_*RELEAVE_START* ${TMPDIR}SYS_ASSEMBLE_START*;

    # Step 1.4: solve the clash between ligand and water
    OTHERNr=$(grep HETATM ${TMPDIR}WatEquil.pdb | sed -e /T3P/d -e /${watCode}/d -e /${aniCode}/d -e /${catCode}/d | wc -l | awk '{print $1}');
    if [[ ${ifreconly} != "true" ]]; then 
        # Use coordinate to find conflict;
        egrep "OW.*T3P|OW.*SPC|OW.*T4P|OW.*T4E|OW.*T5P" ${TMPDIR}WatEquil.pdb | awk '{print substr($0,7,5), substr($0,23,4), substr($0,31,8), substr($0,39,8), substr($0,47,8)}'| awk '{print $1,$2,$3,$4,$5}' > ${TMPDIR}wat_str.cor
        grep OW.*T3P ${pdbfile} | awk '{print substr($0,7,5), substr($0,23,4), substr($0,31,8), substr($0,39,8), substr($0,47,8)}' | while read i; do re=$(echo ${i} | tr " " "%"|sed 's|%|\.\*|g'); sed -i "/${re}/d" ${TMPDIR}wat_str.cor; done
        tmpNr=$(cat -n ${ligfile} | grep '<TRIPOS>MOLECULE'| awk '{print $1+2}');
        ligAtomNr=$(sed -n ${tmpNr}p ${ligfile} | awk '{print $1}'); 
        grep ATOM -A${ligAtomNr} ${ligfile} | sed /ATOM/d| awk '{print $3,$4,$5}' > ${TMPDIR}/lig_str.cor; 


        # Get the OTHER atoms to Ligand category
        if [ ${OTHERNr} -gt 0 ]; then 
            echo "${OTHERNr} OTHER atoms identified, including them to LIGAND category;"
            grep HETATM ${TMPDIR}WatEquil.pdb | sed -e /T3P/d -e /${watCode}/d -e /${aniCode}/d -e /${catCode}/d | awk '{print substr($0, 31, 24)}' >> ${TMPDIR}/lig_str.cor; 
        fi 
        python3 -c """${find_clash_script}""" ${TMPDIR}wat_str.cor ${TMPDIR}lig_str.cor LIG 2.5 ${TMPDIR}ligclashes.cor;
        if ! [ -f ${TMPDIR}ligclashes.cor ]; then echo "Python script for clash solving did not work properly"; exit 11; fi 
        python3 -c """${rmClashWat}""" ${TMPDIR}ligclashes.cor ${TMPDIR}WatEquil.pdb ${TMPDIR}rec_solved2.pdb && echo "find_clash_script/rmClashWat: solving water clashes (Ligand)";
    elif [[ ${ifreconly} == "true" ]]; then 
        if [ ${OTHERNr} -gt 0 ]; then
            echo "${OTHERNr} OTHER atoms identified, including them to LIGAND category;";
            grep HETATM ${TMPDIR}WatEquil.pdb | sed -e /T3P/d -e /${watCode}/d -e /${aniCode}/d -e /${catCode}/d | awk '{print substr($0, 31, 24)}' >> ${TMPDIR}/lig_str.cor;
            python3 -c """${find_clash_script}""" ${TMPDIR}wat_str.cor ${TMPDIR}lig_str.cor LIG 2.5 ${TMPDIR}ligclashes.cor;
            if ! [ -f ${TMPDIR}ligclashes.cor ]; then echo "Python script for clash solving did not work properly"; exit 11; fi
            python3 -c """${rmClashWat}""" ${TMPDIR}ligclashes.cor ${TMPDIR}WatEquil.pdb ${TMPDIR}rec_solved2.pdb && echo "find_clash_script/rmClashWat: solving water clashes (Ligand)";
        else
            cp ${TMPDIR}WatEquil.pdb ${TMPDIR}rec_solved2.pdb;
        fi
    fi 

    fileCheck ${TMPDIR}rec_solved2.pdb "Structure output after solving clash with ligand" 24;
    rm -f ${TMPDIR}lig_str.cor ${TMPDIR}wat_str.cor;

    spliter "GROMACS: Converting protein PDB to Gromacs format $(date +%b.%d.%y-%H:%M)";

    # Step 2.1: GROMACS Convert protein from PDB format to Gromacs format
    # determine disulfide bonds number and chain number 
    dsfbondNr=$( cat -n  ${REFDIR}singleMD/target_wat_ion.seq | awk '{if (($2 == "CYS" || $2 == "CYX") && ($3 != "")){print $2,$1"-"$3";"}}' | wc -l );
    dsfbonds=$( cat -n  ${REFDIR}singleMD/target_wat_ion.seq | awk '{if (($2 == "CYS" || $2 == "CYX") && ($3 != "")){print $2,$1"-"$3";"}}'); 
    dfsinputStr=$(tmpstr=""; for i in $(seq 1 ${dsfbondNr}); do tmpstr=${tmpstr}"y\n"; if [ ${i} -eq ${dsfbondNr} ]; then echo ${tmpstr}; fi ; done) 
    if [ ${dsfbondNr} -gt 0 ]; then echo -e "${dsfbondNr} disulfide bonds are detected and they are:\n${dsfbonds}";
    else echo -e "${dsfbondNr} disulfide bonds are detected"
    fi 
    grep ATOM ${TMPDIR}rec_solved2.pdb > ${TMPDIR}TopGen_pro.pdb;

    # TODO: tmp solution. 
    sed -i "s|NAC|NME|g" ${TMPDIR}TopGen_pro.pdb; 

    chainNr=$(grep ATOM ${TMPDIR}TopGen_pro.pdb | awk '{print substr($0, 22, 1)}' | sed /^$/d | uniq | wc -l );

    breaks=$(python3 -c """${breakDetect}""" ${TMPDIR}TopGen_pro.pdb);
    breakNr=$(echo ${breaks} | tr '%' '\n' | sed /^$/d | wc -l | awk '{print $1+1}');   # +1 because there is the start and the end terminal.
    echo "There are ${breakNr} breaks: ${breaks}";
    tmpinput=${dfsinputStr}; 
    if [ ${breakNr} -gt 0 ]; then  for i in $(seq 1 ${breakNr}); do tmpinput=${tmpinput}"0\n0\n"; done; fi
    echo "Test with the following input string ${tmpinput}" 2>&1;

    # Step 2.2 Prepare protein topology and protein part 
    rm -f ${TMPDIR}topol.top ${TMPDIR}rec_primary.gro;
    echo -e ${topTemplate} | sed "s|THESYSNAME|EXPSIM|g" | tr '%' '\n' > ${TMPDIR}topol.top; 

    if [ ${breakNr} -gt 1 ]; then    # Multi-chain mode
        # add TER record at the end of each chains and add END record at the end of PDB;
        # assemble different chains -> convert the topfile to itp -> form a short overall topol.top file 
        grep ATOM ${TMPDIR}TopGen_pro.pdb | awk '{print substr($0, 22, 1)}' | sed /^$/d | uniq| while read i; do terNr=$(cat -n ${TMPDIR}TopGen_pro.pdb | grep " ${i} " | tail -1 | awk '{print $1}'); sed -i "${terNr}a TER" ${TMPDIR}TopGen_pro.pdb; done
        echo -e "END" >> ${TMPDIR}TopGen_pro.pdb;
    elif [ ${breakNr} -eq 1 ]; then    # Single-chain mode
        grep ATOM ${TMPDIR}rec_solved2.pdb > ${TMPDIR}TopGen_pro.pdb;
        echo -e "END" >> ${TMPDIR}TopGen_pro.pdb;
    fi 

    # Make sure the pdb2gmx will not be a zombie process
    echo -e "${tmpinput}" | gmx pdb2gmx -f ${TMPDIR}TopGen_pro.pdb -o ${TMPDIR}protein.gro -ff ${charmmFF} -water ${wat_itp} -ss -chainsep ter -merge all -i ${TMPDIR}posre.itp -p ${TMPDIR}multichain.top -ter > ${TMPDIR}tmp.txt & (sleep 0.01; pid=$(ps aux| grep pdb2gmx | head -1 | awk '{print $2}'); sleep 30; kill -9 ${pid});
    fileCheck ${TMPDIR}tmp.txt "Failed to correctly predict the disulfide bond number or terminal residue number" 42;

    terinputStr=$(python3 -c """${getTerString}""" ${TMPDIR}tmp.txt);
    terinputStr=$(echo ${terinputStr} | sed 's/%/\\n/g')
    pdb2gmxStr=${dfsinputStr}${terinputStr};
    echo "The final input string for pdb2gmx is ${pdb2gmxStr}"
    echo -e "${pdb2gmxStr}" | gmx pdb2gmx -f ${TMPDIR}TopGen_pro.pdb -o ${TMPDIR}protein.gro -ff ${charmmFF} -water ${wat_itp} -ss -ter -chainsep ter -merge all -i ${TMPDIR}posre.itp -p ${TMPDIR}proteinchain.top -nocmap >> ${TMPDIR}GMX_Executions.log 2>&1 & (sleep 0.01; pid=$(ps aux| grep pdb2gmx | head -1 | awk '{print $2}'); sleep 30; kill -9 ${pid}); 

    fileCheck ${TMPDIR}protein.gro "Gromacs topology/coordinate" 31; 
    fileCheck ${TMPDIR}proteinchain.top "Gromacs topology/coordinate" 31;
 
    typeNr=$(cat -n ${TMPDIR}proteinchain.top | grep "moleculetype" | awk '{print $1+2}');  typeStr=$(sed -n ${typeNr}p ${TMPDIR}proteinchain.top | awk '{print  "Protein        "$2}');
    sed -i "${typeNr}s|.*|${typeStr}|" ${TMPDIR}proteinchain.top;
    itpend=$(cat -n ${TMPDIR}proteinchain.top | grep 'Include Position restraint file' | awk '{print $1-1}');
    sed -n 1,${itpend}p ${TMPDIR}proteinchain.top > ${TMPDIR}proteinchain.itp;
    sed -n /^.*include.*${charmmFF}.*$/p ${TMPDIR}proteinchain.top > ${TMPDIR}defff.txt;  # avoid redefine of topology and parameters in the forcefield 
    sed -i /^.*include.*${charmmFF}.*$/d ${TMPDIR}proteinchain.itp;

    incStr="\#include \".\/proteinchain.itp\"";
    incind=$(cat -n ${TMPDIR}topol.top | grep "Include protein topology" | awk '{print $1+1}');
    molStr="Protein        1";
    molind=$(cat -n ${TMPDIR}topol.top | grep "\[ molecules \]" | awk '{print $1+2}');
    sed -i -e "${incind}i  ${incStr}" -e "${molind}i ${molStr}" ${TMPDIR}topol.top;
    posre_index=$(cat -n ${TMPDIR}topol.top | grep 'Include position restraint file' | awk '{print $1+1}'); 
    posre_str="#ifdef POSRES\n#include \".\/posre.itp\"\n#endif";
    sed -i -e "${posre_index}i  ${posre_str}"  ${TMPDIR}topol.top;

    mv ${TMPDIR}protein.gro  ${TMPDIR}rec_primary.gro; 
    fileCheck ${TMPDIR}defff.txt "Gromacs parameter information" 9;
    fileCheck ${TMPDIR}topol.top "Gromacs topology file" 9;
    fileCheck ${TMPDIR}rec_primary.gro "Gromacs structure file" 9;

    ffStr=$(grep "forcefield.itp" ${TMPDIR}defff.txt); ffind=$(cat -n ${TMPDIR}topol.top | grep "Include forcefield parameters" | awk '{print $1+1}');
    watStr=$(egrep "tip3p.itp|spc.itp" ${TMPDIR}defff.txt); watind=$(cat -n ${TMPDIR}topol.top | grep "Include water topology" | awk '{print $1+1}');
    ionStr=$(grep "ions.itp" ${TMPDIR}defff.txt); ionind=$(cat -n ${TMPDIR}topol.top | grep "Include topology for ions" | awk '{print $1+1}');
    if [ ${ffind} -gt 0 ] || [ ${watind} -gt 0 ] || [ ${ionind} -gt 0 ]; then   sed -i -e "${ffind}i ${ffStr}" -e "${watind}i ${watStr}" -e "${ionind}i ${ionStr}" ${TMPDIR}topol.top;
    else  echo "Warning: failed to correctly prepare the topology file" 1>&2; exit 9;
    fi 

    # Step 2.3 Add Ligand topology to the topol.top
    if [[ ${ifreconly} != "true" ]]; then
        # Note: Ligand topology preparation and output file check already done in the previous step 
        if [[ $(grep 'ATOM' ${TMPDIR}MDL.str | wc -l) -ne $(grep 'HETATM' ${TMPDIR}LIG_CAMPOrder.pdb | wc -l) ]] ; then echo "Fatal: cgenff did not correctly generate atom" 1>&2; exit 32; fi
        ligMatchStr=$(python3 -c """${matchLIGIndex}""" ${TMPDIR}LIG_CAMPOrder.pdb ${TMPDIR}mdl_ini.pdb LIG MDL) && echo "python->matchLIGIndex: Matching the atom orders between CAMPARI ligand/Gromacs ligand"; 
        if [ ${#ligMatchStr} -eq 0 ]; then echo "Ligand atoms failed to be matched between CAMPARI and Gromacs" 1>&2; exit 22; fi 
        lig_top_index=$(cat -n  ${TMPDIR}topol.top | grep 'Include ligand topology' | awk '{print $1+1}'); LIG_top="#include \".\/mdl.itp\"\n"; 
        lig_prm_index=$(cat -n  ${TMPDIR}topol.top | grep 'Include ligand parameters' | awk '{print $1+1}'); LIG_parm="#include \".\/mdl.prm\"\n";
        lig_posre_index=$(cat -n ${TMPDIR}topol.top | grep 'Include ligand position restraints' | awk '{print $1+1}'); LIG_posre="#ifdef POSRES\n#include \".\/posre_mdl.itp\"\n#endif\n";
        sed -i -e "${lig_top_index}i  ${LIG_top}" -e "${lig_prm_index}i ${LIG_parm}" -e "${lig_posre_index}i ${LIG_posre}" ${TMPDIR}topol.top; 
    fi

    # Step 2.4: Extract and prepare OTHER molecules;
    # Other atom: All atoms that not belong to protein, ligand, waters and ions
    if [ ${OTHERNr} -gt 0 ]; then 
        rm -f ${TMPDIR}other* ${TMPDIR}tmp.txt;
        grep HETATM ${TMPDIR}rec_solved2.pdb | sed -e /T3P/d -e /${watCode}/d -e /${aniCode}/d -e /${catCode}/d > ${TMPDIR}other_mols.pdb;
        gmx -quiet -nobackup editconf -f ${TMPDIR}other_mols.pdb -o ${TMPDIR}other_mols.gro >> ${TMPDIR}GMX_Executions.log 2>&1;
        sed -i -e 1d -e 2d -e '$d' ${TMPDIR}other_mols.gro  && mv ${TMPDIR}other_mols.gro ${TMPDIR}other_mols.str; 
        cat ${TMPDIR}other_mols.pdb | awk '{print substr($0,23,4), substr($0,18,3)}'| uniq | awk '{print $2}'| sort -n -k2 | uniq -c | awk '{print $2"    "$1}' > ${TMPDIR}other_items.txt;
        rm -f ${TMPDIR}other_mols.pdb;

        python3 -c """${cofactorCAMP2GMX}""" ${TMPDIR}other_items.txt ${TMPDIR}tmp.txt && mv ${TMPDIR}tmp.txt ${TMPDIR}other_items.txt; 
        python3 -c """${cofactorCAMP2GMX}""" ${TMPDIR}other_mols.str ${TMPDIR}tmp.txt && mv ${TMPDIR}tmp.txt ${TMPDIR}other_mols.str; 
    fi 

    # Step 2.5: Extract solvent waters and ions;
    egrep " ${watCode} | T3P " ${TMPDIR}rec_solved2.pdb > ${TMPDIR}solvent_ion.pdb; 
    egrep " ${catCode} " ${TMPDIR}rec_solved2.pdb >> ${TMPDIR}solvent_ion.pdb; 
    egrep " ${aniCode} " ${TMPDIR}rec_solved2.pdb >> ${TMPDIR}solvent_ion.pdb; 
    egrep "${catCode}|${aniCode}|${watCode}|T3P" ${TMPDIR}solvent_ion.pdb | awk '{print substr($0,23,4), substr($0,18,3)}' | uniq | awk '{print $2}'| sort -n -k2 | uniq -c | awk '{print $2"    "$1}' > ${TMPDIR}wat_ion_items.txt; 

    gmx -quiet -nobackup editconf -f ${TMPDIR}solvent_ion.pdb -o ${TMPDIR}solvent_ion.gro  >> ${TMPDIR}GMX_Executions.log 2>&1;
    sed -i -e 1d -e 2d -e '$d' ${TMPDIR}solvent_ion.gro && mv ${TMPDIR}solvent_ion.gro ${TMPDIR}solvent_ion.str;
    sed -i -e "s/${catCode}/${catGMX}/g" -e "s/${aniCode}/${aniGMX}/g" -e "s/${watCode}\|T3P/SOL/g" ${TMPDIR}solvent_ion.str;
    sed -i -e "s/${catCode}/${catGMX}/g" -e "s/${aniCode}/${aniGMX}/g" -e "s/${watCode}\|T3P/SOL/g" ${TMPDIR}wat_ion_items.txt; 
    fileCheck ${TMPDIR}solvent_ion.str "Solvent/Ion gro structure" 9;
    fileCheck ${TMPDIR}wat_ion_items.txt "Solvent/Ion item list" 9;
    # rm -f  ${TMPDIR}solvent_ion.pdb  ${TMPDIR}solvent_ion.gro;

    # Step 2.6: Count different components in the system and insert into topology
    PN=$(sed -n 2p ${TMPDIR}rec_primary.gro | tr -d ' ' | awk '{print $1}');
    SOL=$(cat ${TMPDIR}wat_ion_items.txt | egrep "SOL" | awk '{print $2}'); SOL_comp=$(cat ${TMPDIR}wat_ion_items.txt | grep SOL);
    CN=$(cat ${TMPDIR}wat_ion_items.txt | egrep "${catGMX}" | awk '{print $2}'); CAT_comp=$(cat ${TMPDIR}wat_ion_items.txt | grep ${catGMX});
    AN=$(cat ${TMPDIR}wat_ion_items.txt | egrep "${aniGMX}" | awk '{print $2}'); ANI_comp=$(cat ${TMPDIR}wat_ion_items.txt | grep ${aniGMX});
    ON=${OTHERNr};
    TN=$(echo ${PN}+$(wc -l ${TMPDIR}solvent_ion.str | awk '{print $1}')+${ON} | bc );

    if [[ ${ifreconly} != "true" ]]; then  
        LIG_index=$(cat -n  ${TMPDIR}topol.top | egrep 'Protein +[0-9]' | awk '{print $1+1}'); 
        LIG_comp="MDL    1"; sed -i -e "${LIG_index}i ${LIG_comp}" ${TMPDIR}topol.top; 
        LN=$(grep  MDL  ${TMPDIR}smdlig.str | wc -l); TN=$(echo ${TN} + ${LN} | bc ); 
    else LN=0;
    fi 
    echo total number is ${TN} PN ${PN} LN ${LN} ON ${ON} SOL ${SOL} AN ${AN} CN ${CN};

    MOLE_index=$(cat -n  ${TMPDIR}topol.top | egrep 'Protein +[0-9]' | awk '{print $1+1}'); 
    sed -i -e "${MOLE_index}i  ${SOL_comp}" -e "${MOLE_index}i  ${CAT_comp}" -e "${MOLE_index}i  ${ANI_comp}" ${TMPDIR}topol.top; 
    if [ ${OTHERNr} -gt 0 ]; then sed -i "$(echo ${MOLE_index}|awk '{print $1-1}')r ${TMPDIR}other_items.txt" ${TMPDIR}topol.top; fi
    if [ ${#PN} -eq 0 ] || [ ${#SOL} -eq 0 ] || [ ${#CN} -eq 0 ] || [ ${#AN} -eq 0 ] || [ ${#TN} -eq 0 ] || [ ${#LN} -eq 0 ]; then echo "Fatal: Components are not well prepared " 1>&2; exit 9; fi

    # Step 2.7:  Assembling gro information of all components and center the box
    cp ${TMPDIR}rec_primary.gro ${TMPDIR}rec_comb.gro;
    sed -i 2d ${TMPDIR}rec_comb.gro && sed -i "2i ${TN}" ${TMPDIR}rec_comb.gro;
    groind=$(cat ${TMPDIR}rec_comb.gro | wc -l | awk '{print $1-1}');
    if [[ ${ifreconly} == "false" ]]; then sed -i "${groind}r ${TMPDIR}smdlig.str" ${TMPDIR}rec_comb.gro; fi
    sed -i "${groind}r ${TMPDIR}solvent_ion.str" ${TMPDIR}rec_comb.gro;
    if [ ${OTHERNr} -gt 0 ]; then sed -i "${groind}r ${TMPDIR}other_mols.str" ${TMPDIR}rec_comb.gro; fi
    gmx_box=$(echo ${new_box} | awk '{print $1*0.1, $2*0.1, $3*0.1}');
    gmx editconf -f ${TMPDIR}rec_comb.gro -o ${TMPDIR}centered_comb.gro -c -box ${gmx_box} -bt cubic -resnr 1 >> ${TMPDIR}GMX_Executions.log 2>&1;
    fileCheck ${TMPDIR}centered_comb.gro 'Input for energy minimization' 9;


    # Step 3.1: Energy minimization by GROMACS   
    echo -e ${em_mdp} | tr "%" "\n" | sed -e "s|EM_NRSTEPS|1000|g" -e "s|GMX_MINI_MODE|steep|g" > ${TMPDIR}em.mdp; 
    gmx grompp -f ${TMPDIR}em.mdp -c ${TMPDIR}centered_comb.gro -r ${TMPDIR}centered_comb.gro -p ${TMPDIR}topol.top -po em_out.mdp -o ${TMPDIR}em.tpr  -maxwarn  100  -nocopyright  >  ${TMPDIR}emlog.txt 2>&1;

    c=0
    while true; do 
        c=$(echo ${c}+1|bc); 
        teststr=$(grep -B1 'Encountered a second block of parameters' ${TMPDIR}emlog.txt) ; 
        echo "This is the test of ERROR in EM.  ${#teststr} \n ${teststr} "
        if [ ${#teststr} -gt 0 ]; then 
            grep -B1 'Encountered a second block of parameters' ${TMPDIR}emlog.txt | sed -e /Encountered/d -e /--/d | sed -e 's|^.*line| |g' -e 's|].*$||g' | awk '{print $1}' | tac | while read i; do sed -i ${i}d ${TMPDIR}mdl.prm; done
            gmx grompp -f ${TMPDIR}em.mdp -c ${TMPDIR}centered_comb.gro -r ${TMPDIR}centered_comb.gro -p ${TMPDIR}topol.top -po em_out.mdp -o ${TMPDIR}em.tpr  -maxwarn  100  -nocopyright  >  ${TMPDIR}emlog.txt 2>&1; 
        elif [ ${c} -gt 10 ] || [ ${#teststr} -eq 0 ]; then
            break
        fi 
    done

    fileCheck ${TMPDIR}em.tpr "Protable simulation file from grompp" 40; 

    spliter "Energy minimization started at $(date +%b.%d.%y-%H:%M)";
    for i in $(seq 6 -1 3); do
        MPINum=$(echo "scale=0; 2^${i}"| bc); OMPNum=$(echo "scale=0; ${SLURM_CPUS_PER_TASK}/${MPINum}" | bc); npme=$(echo "scale=0; ${MPINum}/4" | bc );
        srun -v -n 1 --cpus-per-task ${SLURM_CPUS_PER_TASK} --ntasks-per-core ${SLURM_NTASKS_PER_CORE}  --mpi=pmix_v2  gmx mdrun -s ${COPYDIR}em.tpr -deffnm em -nt ${SLURM_CPUS_PER_TASK} -ntmpi ${MPINum}  -ntomp ${OMPNum} -npme ${npme} -nocopyright  >> ${TMPDIR}emlog.txt 2>&1; 
        if [ -f ${COPYDIR}em.gro ]; then break; fi
    done
    spliter "Energy minimization finished at $(date +%b.%d.%y-%H:%M)";
    # srun -v -n 1 --cpus-per-task ${SLURM_CPUS_PER_TASK} --ntasks-per-core ${SLURM_NTASKS_PER_CORE}  --mpi=pmix_v2  gmx  mdrun -s ${TMPDIR}em.tpr  -deffnm  em  -nt ${SLURM_CPUS_PER_TASK}  -ntmpi  ${SLURM_CPUS_PER_TASK}  -ntomp 1  -npme ${pmeNr}  -nsteps -2  -nocopyright  >> ${TMPDIR}emlog.txt 2>&1; 


    echo 'q' | gmx make_ndx -f ${TMPDIR}em.gro > ${TMPDIR}tmp.txt 2>> ${TMPDIR}GMX_Executions.log;
    if [[ ${ifreconly} == "false" ]]; then 
        mdl_groupNr=$(grep MDL ${TMPDIR}tmp.txt | head -1 | awk '{print $1}');
        echo -e "1|${mdl_groupNr} \n q" | gmx make_ndx -f ${TMPDIR}em.gro -o ${TMPDIR}rec_lig_index.ndx >> ${TMPDIR}GMX_Executions.log 2>&1;
    else  
        echo 'q' | gmx make_ndx -f ${TMPDIR}em.gro -o ${TMPDIR}rec_lig_index.ndx >> ${TMPDIR}GMX_Executions.log 2>&1; 
    fi 

    fileCheck ${TMPDIR}rec_lig_index.ndx "Index file of the system." 33 ;
    fileCheck ${TMPDIR}em.gro "Energy minimization structure" 34;

    # Step 3.2: NPT equilibration by GROMACS
    echo -e ${npt_mdp_equil} | tr "%" "\n" | sed -e "s|GMXNPTSTEP|${nptsteps}|g" -e "s|GMXTIMESTEP|${timestep}|g" -e "s|GMXOUTFREQ|500|g" -e "s|GMXPRESS|${pressure}|g" -e "s|GMXTEMP|${temperature}|g" -e "s|RESTMODE|-DPOSRES|g" > ${TMPDIR}npt.mdp; 
    if [[ ${ifreconly} == "false" ]]; then sed -i -e "s|GMXPROGROUP|Protein_MDL|g" ${TMPDIR}npt.mdp;
    else sed -i -e "s|GMXPROGROUP|Protein|g" ${TMPDIR}npt.mdp; 
    fi

    gmx grompp -f ${TMPDIR}npt.mdp -c ${TMPDIR}em.gro -r ${TMPDIR}em.gro -p ${TMPDIR}topol.top -po npt_out.mdp -o ${TMPDIR}npt.tpr  -n ${TMPDIR}rec_lig_index.ndx -maxwarn 100 -nocopyright > ${TMPDIR}nptlog.txt 2>&1;
    fileCheck ${TMPDIR}npt.tpr "Protable simulation file from grompp" 40; 

    spliter "NPT equilibration started at $(date +%b.%d.%y-%H:%M)"; 
    for i in $(seq 6 -1 3); do
        MPINum=$(echo "scale=0; 2^${i}"| bc); OMPNum=$(echo "scale=0; ${SLURM_CPUS_PER_TASK}/${MPINum}" | bc); npme=$(echo "scale=0; ${MPINum}/4" | bc );
        srun -v -n 1 --cpus-per-task ${SLURM_CPUS_PER_TASK} --ntasks-per-core ${SLURM_NTASKS_PER_CORE}  --mpi=pmix_v2  gmx mdrun -s ${COPYDIR}npt.tpr -deffnm npt -nt ${SLURM_CPUS_PER_TASK} -ntmpi ${MPINum}  -ntomp ${OMPNum} -npme ${npme} -nocopyright  >> ${TMPDIR}nptlog.txt 2>&1;
        if [ -f ${COPYDIR}npt.gro ]; then break; fi
    done
    spliter "NPT equilibration finished at $(date +%b.%d.%y-%H:%M)";
    # srun -v -n 1 --cpus-per-task ${SLURM_CPUS_PER_TASK} --ntasks-per-core ${SLURM_NTASKS_PER_CORE}  --mpi=pmix_v2 gmx mdrun -s ${TMPDIR}npt.tpr -deffnm npt -nt ${SLURM_CPUS_PER_TASK} -ntmpi ${SLURM_CPUS_PER_TASK} -ntomp 1 -npme ${pmeNr} -nsteps -2 -nocopyright  >> ${TMPDIR}nptlog.txt 2>&1; 


    npterr1=$(grep "This usually means that your system is not well equilibrated" ${TMPDIR}npt.log);
    if [ ${#npterr1} -gt 1 ]; then echo "Fatal: The System is not well equilibrated" 1>&2 ; exit 38; fi
    npterr2=$(grep 'Finished mdrun' ${TMPDIR}npt.log); emerr1=$(grep 'Finished mdrun' ${TMPDIR}em.log); 
    if [ ${#npterr2} -eq 0 ] || [ ${#emerr1} -eq 0 ]; then echo "Fatal: Energy minimization or NPT equilibration exit unexpectedly." 1>&2; exit 35; fi

    if [[ ${ifreconly} == "false" ]]; then
        indexerr1=$(grep 'Protein_MDL' ${TMPDIR}rec_lig_index.ndx);
        if [ ${#indexerr1} -eq 0 ]; then echo "FataL: fails to find the Protein_MDL group in index file." 1>&2; exit 33; fi 
    fi 
    nptsize=$(tail -1 ${TMPDIR}npt.gro | awk '{print $1*10"_"$2*10"_"$3*10}');

    fileCheck ${TMPDIR}npt.gro "NPT output structure" 35;
    fileCheck ${TMPDIR}npt.trr "NPT trajectory output" 35;  
    fileCheck ${TMPDIR}npt.cpt "NPT check point file" 35;

    # Start to collect log files 
    rm -f ${TMPDIR}checks.log; 
    egrep "srun.*id|srun.*task|srun.*user|----|srun.*job|srun.*bind" ${TMPDIR}nptlog.txt | sed -e /N/d -e /null/d > ${TMPDIR}checks.log;
    spliter "CAMPARI System assemble" >> ${TMPDIR}checks.log; 
    endcheck=$(sed -n '/SUMMARY OF CALCULATION/,$p' ${TMPDIR}System_Assemble.log);
    if [ ${#endcheck} -gt 0 ]; then  sed -n -e '/SUMMARY OF CALCULATION/, /END OF SUMMARY/p' -e '/Total CPU time elapsed/, $ p' ${TMPDIR}System_Assemble.log >> ${TMPDIR}checks.log;
    else  egrep -i -e "Warning|Fatal" ${TMPDIR}System_Assemble.log >> ${TMPDIR}checks.log; 
    fi 

    spliter "CAMPARI NVT simulation" >> ${TMPDIR}checks.log;
    endcheck=$(sed -n '/SUMMARY OF CALCULATION/,$p' ${TMPDIR}NVT_log.txt); 
    if [ ${#endcheck} -gt 0 ]; then  sed -n -e '/SUMMARY OF CALCULATION/, /END OF SUMMARY/p' -e '/Total CPU time elapsed/, $ p' ${TMPDIR}NVT_log.txt >> ${TMPDIR}checks.log;
    else  egrep -i -e "Warning|Fatal" ${TMPDIR}NVT_log.txt >> ${TMPDIR}checks.log;
    fi 

    spliter "GROMACS energy minimization" >> ${TMPDIR}checks.log;
    endcheck=$(sed -n "/Energy minimization/,$ p" ${TMPDIR}em.log );
    if [ ${#endcheck} -gt 0 ]; then  sed -n -e "/Running on.*node with total/, /PLEASE READ AND CITE/p" -e '/There are.*Atoms/,/Started mdrun on.*/p' -e "/Energy minimization/,$ p"  ${TMPDIR}em.log | sed '/PLEASE READ AND CITE/d'  >> ${TMPDIR}checks.log;
    else   tail -30 ${TMPDIR}em.log >> ${TMPDIR}checks.log; 
    fi 

    spliter "GROMACS NPT equilibration" >> ${TMPDIR}checks.log;
    endcheck=$(sed -n "/A V E R A G E S/, $ p" ${TMPDIR}npt.log);
    if [ ${#endcheck} -gt 0 ]; then  sed -n -e '/Running on.*node with total/, /PLEASE READ AND CITE/p' -e '/Initializing Domain Decomposition/, /PLEASE READ AND CITE/p' -e '/There are.*Atoms/,/Started mdrun on.*/p' -e "/A V E R A G E S/, $ p"  ${TMPDIR}npt.log | sed '/PLEASE READ AND CITE/d'  >>  ${TMPDIR}checks.log;
    else  tail -30 ${TMPDIR}npt.log  >>  ${TMPDIR}checks.log; 
    fi 

    # Step 3.3: Get the ligand atom re-ordered index (CAMPARI preferred convention)
    if [[ ${ifreconly} != "true" ]]; then
        gmx -quiet editconf -f ${TMPDIR}npt.gro -o ${TMPDIR}NPT_MKNDX.pdb >> ${TMPDIR}GMX_Executions.log 2>&1;
        startIND=$(grep MDL ${TMPDIR}NPT_MKNDX.pdb | head -1| awk '{print substr($0,7,5)}' | awk '{print $1}'); 
        orderLig=$(echo ${ligMatchStr} | tr '%' '\n'| sed /^$/d | tr '-' ' ' | awk -vinc=${startIND} '{print $2+inc-1}' | tr '\n' '%');
        foundNr=$(echo ${orderLig} | tr '%' '\n' | sed /^$/d | wc -l );
        if [ ${foundNr} -ne $(grep "HETATM" ${TMPDIR}LIG_CAMPOrder.pdb | wc -l | awk '{print $1}') ]; then echo "There is mismatch between CAMPARI initial ligand atom order and matched atoms" 1>&2; exit 22;  fi 
        python3 -c """${getCAMPProLigNDX}""" ${TMPDIR}lig_reorder.ndx ${startIND} ${orderLig} && echo 'getCAMPProLigNDX: generating the NDX file for sequence re-order'
        fileCheck ${TMPDIR}lig_reorder.ndx "Re-ordered index file" 14;
        cat ${TMPDIR}lig_reorder.ndx >> ${TMPDIR}rec_lig_index.ndx;
    fi 

    # Step 3.4: Prepare the sequence file for trajectory conversion and production execution script
    cp ${REFDIR}tmp_Sampling_target.seq ${TMPDIR}trj_conv.seq;
    sed -i -e /T3P/d -e /${watCode}/d -e /${aniCode}/d -e /${catCode}/d ${TMPDIR}trj_conv.seq;
    WatStr3=$(getSolStr ${SOL} ${watCode}); cationStr=$(getSolStr ${CN} ${catCode}); anionStr=$(getSolStr ${AN} ${aniCode});
    insertStr3=${WatStr3}${cationStr}${anionStr};
    echo ${insertStr3} > ${TMPDIR}tmp.txt; 
    if [[ ${ifreconly} == "true" ]]; then  i_index=$(cat -n  ${TMPDIR}trj_conv.seq | grep END | awk '{print $1-1}'); 
    elif [[ ${ifreconly} != "true" ]]; then i_index=$(cat -n ${TMPDIR}trj_conv.seq | grep LIG_N_C | awk '{print $1-1}');
    fi 
    sed -i "${i_index}r ${TMPDIR}tmp.txt" ${TMPDIR}trj_conv.seq && sed -i "s/%/\n/g" ${TMPDIR}trj_conv.seq && sed -i /^$/d ${TMPDIR}trj_conv.seq;
    if [[ ${ifreconly} == "true" ]]; then sed -i /LIG_N_C/d ${TMPDIR}trj_conv.seq ; fi
    fileCheck ${TMPDIR}trj_conv.seq "Sequence file for the trajecotry conversion" 25;

    if [ ${SLURM_NTASKS_PER_CORE} -eq 1 ]; then 
        cpumask=$(/storage006/yzhang/tests/MaskGen | egrep -A1 "${SLURM_NTASKS}x.*${SLURM_CPUS_PER_TASK}.*physical cores"| tail -1 | sed 's/,$//');
    elif [ ${SLURM_NTASKS_PER_CORE} -eq 2 ]; then 
        cpumask=$(/storage006/yzhang/tests/MaskGen | egrep -A1 "${SLURM_NTASKS}x.*${SLURM_CPUS_PER_TASK}.*HT assuming logical cores"| tail -1 | sed 's/,$//'); 
    else 
        cpumask="";
    fi 
    echo "cpu mast is as following: ${cpumask}"; 

    rm -rf ${REFDIR}singleMD/scratch; 
    mkdir -p ${REFDIR}singleMD/scratch;
    cd ${TMPDIR}
    cp -r topol.top rec_lig_index.ndx mdl.prm mdl.itp MDL.str posre_mdl.itp posre.itp ${charmmDirName} npt.cpt npt.gro mdl_ini.pdb proteinchain.itp trj_conv.seq ${REFDIR}singleMD/scratch/

    cat ${SCRIPTDIR}smd_initprod.sh > ${REFDIR}singleMD/testprod.sh;
    echo "TMPDIR=${TMPDIR}%REFDIR=${REFDIR}%SCRIPTDIR=${SCRIPTDIR}%CAMPDATADIR=${CAMPDATADIR}%CAMPPARM=${CAMPPARM}%CAMPEXE_NOMPI=${CAMPEXE_NOMPI}%CAMPMPI_MPI_THREADEXE=${CAMPMPI_MPI_THREADEXE}%prodeng=${prodeng}%nrsteps=${nrsteps}%timestep=${timestep}%outfreq=${outfreq}%pressure=${pressure}%temperature=${temperature}%ifreconly=${ifreconly}%outputformat=${outputformat}%simmode=${simmode}%trajext=${trajext}%ensemble=${ensemble}%forcefield=${forcefield}%neuiontype=${neuiontype}%outputgroup=${outputgroup}%nptsize=${nptsize}%errorlog=${TMPDIR}checks.log%lastNr=${lastNr}%parmfile=${parmfile}%ligfile=${ligfile}%slotsel=${slotsel}%wat_itp=${wat_itp}%charmmDirName=${charmmDirName}" | tr '%' '\n' > ${TMPDIR}tmp.txt;
    echo "LigMap=${ligMatchStr}" >> ${TMPDIR}tmp.txt; 
    insertInd=$(cat -n ${REFDIR}singleMD/testprod.sh | grep INSERT_VARIABLE_HERE | awk '{print $1}'); 
    sed -i "${insertInd}r  ${TMPDIR}tmp.txt" ${REFDIR}singleMD/testprod.sh;

    # Step 3.5: Execute the scripts 
    spliter "Explicit simulation script starts at $(date +%b.%d.%y-%H:%M)";
    chmod +x ${REFDIR}singleMD/testprod.sh; 
    srun --wait=7200 -v -n ${nrcopy} --cpus-per-task=${SLURM_CPUS_PER_TASK} --ntasks-per-core=${SLURM_NTASKS_PER_CORE} --mpi=pmix_v2 ${REFDIR}singleMD/testprod.sh > ${TMPDIR}EXP_productions.log 2>&1;
    checklast=$(ls -d  ${REFDIR}singleMD/[CTRL][0-9][0-9][0-9]${tmpfolder} | awk -F / '{print $(NF)}' | awk '{print substr($0,2,3)}' | sort -n | tail -1 );
    if [ ${lastNr} -eq ${checklast} ]; then  echo "Fatal: No new Trajectory is detected."; exit 62; fi 
    last_id=$(ls -d  ${REFDIR}singleMD/[CTRL][0-9][0-9][0-9]${tmpfolder} | awk -F / '{print $(NF)}' | sort -nk1.2,1.4 | tail -1 );
    spliter "In the past simulation, the latest generated trajectory ID is ${last_id}";
    spliter "Explicit simulation script ends at $(date +%b.%d.%y-%H:%M)";

    # Step 4.1: final check of the log file. 
    check1=$(egrep "Fatal:|exit code|The designated file.*is not found or too small" ${TMPDIR}EXP_productions.log );
    Exit_code=$(egrep "Fatal:|exit code|The designated file.*is not found or too small" ${TMPDIR}EXP_productions.log | sed "s|.*exit code ||g" | awk '{print $1}'); 
    if [ ${#check1} -ne 0 ]; then  Exit_code=${Exit_code:-54}; exit ${Exit_code}; fi

    check2=$(egrep "Simulation Success" ${TMPDIR}EXP_productions.log | head -1 );
    if [ ${#check2} -ne 0 ]; then  echo "Simulation successed"; exit 1;
    else  echo "No explicit exit code - Fail in later stage"; exit 68; 
    fi 

elif ( [[ ${ifrestart} == "true" ]] && [[ ${ifexp} == 'true' ]] ) ; then
    echo "=======>>>>>>> Restart previous simulation with Gromacs software"; 
    cp -r ${charmmPath} ${TMPDIR}; python3 -c """${patchFF}"""  ${charmmPath}; 
    cp ${SCRIPTDIR}cgenff_charmm2gmx_py3_nx1.py ${REFDIR};
    RSTREFDIR=${REFDIR}singleMD/${rstfrom}/;

    if [ ${solventmodel} == 11 ]; then watCode='T3P'; wat_itp='tip3p';
    elif [ ${solventmodel} == 12 ]; then watCode='SPC'; wat_itp='spc';
    else echo "No valid water model is given in explicit solvent simulation" 1>&2; exit 6;
    fi

    forcefield=charmm36.prm;
    if [[ ${forcefield} == "ff1" ]]; then forcefield=charmm36.prm;
    elif [[ ${forcefield} == "ff2" ]]; then forcefield=oplsaal.prm;
    else forcefield=charmm36.prm;
    fi
    if [[ ${outputformat} == "netcdf" ]]; then  outputformat=5; trajext=nc;
    elif [[ ${outputformat} == "xtc" ]]; then outputformat=4; trajext=xtc;
    elif [[ ${outputformat} == "dcd" ]]; then outputformat=3; trajext=dcd;
    elif [[ ${outputformat} == "pdb" ]]; then outputformat=2; trajext=pdb;
    fi

    if [ -s ${RSTREFDIR}${rstfrom}_RSTPKG.tar ]; then   # Extract restart package
        rm -f ${TMPDIR}checkpoint.cpt ${TMPDIR}checkpoint.gro ${TMPDIR}checkpoint.pdb ${TMPDIR}checkpoint.mol2 ${TMPDIR}checkpoint.seq 
        rm -f ${TMPDIR}rec_lig_index.ndx ${TMPDIR}topol.top ${TMPDIR}MDL.str ${TMPDIR}mdl.prm ${TMPDIR}mdl.itp ${TMPDIR}posre.itp;
        rm -f ${TMPDIR}autoPatch_*;
        tar -xvf ${RSTREFDIR}${rstfrom}_RSTPKG.tar -C ${TMPDIR};
        fileCheck ${TMPDIR}topol.top  "Missing file in restart package " 63;
        fileCheck ${TMPDIR}posre.itp  "Missing file in restart package " 63;
        fileCheck ${TMPDIR}checkpoint.pdb  "Missing file in restart package " 63;
        fileCheck ${TMPDIR}checkpoint.seq  "Missing file in restart package " 63;
        fileCheck ${TMPDIR}rec_lig_index.ndx  "Missing file in restart package " 63;
        fileCheck ${TMPDIR}singleMD_ScriptsVars.sh  "Missing file in restart package " 63;
    else
        echo "No revious restart package is found " 1>&2; exit 10;
    fi 
    if ! [ -d ${TMPDIR}${charmmDirName} ]; then echo "Missing file in restart package " 1>&2; exit 63;  fi 
    if ([[ ${ifreconly} != "true" ]]); then 
        fileCheck ${TMPDIR}checkpoint.mol2  "Missing file in restart package " 63;
        fileCheck ${TMPDIR}posre_mdl.itp  "Missing file in restart package " 63; 
        fileCheck ${TMPDIR}mdl_ini.pdb  "Missing file in restart package " 63;
        fileCheck ${TMPDIR}MDL.str  "Missing file in restart package " 63;
        fileCheck ${TMPDIR}mdl.prm  "Missing file in restart package " 63;
        fileCheck ${TMPDIR}mdl.itp  "Missing file in restart package " 63;
    fi 

    if [ ${preveng} == "gmx" ];then
        fileCheck ${TMPDIR}checkpoint.cpt  "Missing file in restart package " 63;
        fileCheck ${TMPDIR}checkpoint.gro  "Missing file in restart package " 63;
        cat ${SCRIPTDIR}smd_rstprod.sh > ${REFDIR}singleMD/exprst.sh;
        echo "TMPDIR=${TMPDIR}%REFDIR=${REFDIR}%SCRIPTDIR=${SCRIPTDIR}%CAMPDATADIR=${CAMPDATADIR}%CAMPPARM=${CAMPPARM}%CAMPEXE_NOMPI=${CAMPEXE_NOMPI}%CAMPMPI_MPI_THREADEXE=${CAMPMPI_MPI_THREADEXE}%nrsteps=${nrsteps}%timestep=${timestep}%outfreq=${outfreq}%pressure=${pressure}%temperature=${temperature}%ifreconly=${ifreconly}%outputformat=${outputformat}%trajext=${trajext}%preveng=${preveng}%rstfrom=${rstfrom}%ensemble=${ensemble}%forcefield=${forcefield}%seqfile=${RSTREFDIR}${rstfrom}_SEQ.seq%neuiontype=${neuiontype}%solventmodel=${solventmodel}%outputgroup=${outputgroup}%ensemble=${ensemble}%lastNr=${lastNr}%parmfile=${parmfile}%prodsize=${prodsize}%nptsize=${nptsize}%slotsel=${slotsel}%wat_itp=${wat_itp}%charmmDirName=${charmmDirName}" | tr "%" "\n" > ${TMPDIR}tmp.txt

        echo "Restarting previous Gromacs simulation to Gromacs at $(date +%b.%d.%y-%H:%M)"; 
        insertInd=$(cat -n ${REFDIR}singleMD/exprst.sh | grep INSERT_VARIABLE_HERE | head -1 | awk '{print $1}');
        sed -i "${insertInd}r  ${TMPDIR}tmp.txt" ${REFDIR}singleMD/exprst.sh;
        chmod +x ${REFDIR}singleMD/exprst.sh;

        srun --wait=7200 -v  -n ${nrcopy} --cpus-per-task=${SLURM_CPUS_PER_TASK}  --ntasks-per-core=${SLURM_NTASKS_PER_CORE} --mpi=pmix_v2 ${REFDIR}singleMD/exprst.sh > ${TMPDIR}EXP_restarts.log 2>&1;

        checklast=$(ls -d  ${REFDIR}singleMD/[CTRL][0-9][0-9][0-9]${tmpfolder} | awk -F / '{print $(NF)}' | awk '{print substr($0,2,3)}' | sort -n | tail -1 );
        if [ ${lastNr} -eq ${checklast} ]; then  echo "Fatal: No new Trajectory is detected."; exit 62; fi
    elif [ ${preveng} == "camp" ];then 
        fileCheck ${TMPDIR}npt.gro  "Missing file in restart package " 63;
        cat ${SCRIPTDIR}smd_rstprod.sh > ${REFDIR}singleMD/exprst.sh;
        echo "TMPDIR=${TMPDIR}%REFDIR=${REFDIR}%SCRIPTDIR=${SCRIPTDIR}%CAMPDATADIR=${CAMPDATADIR}%CAMPPARM=${CAMPPARM}%CAMPEXE_NOMPI=${CAMPEXE_NOMPI}%CAMPMPI_MPI_THREADEXE=${CAMPMPI_MPI_THREADEXE}%nrsteps=${nrsteps}%timestep=${timestep}%outfreq=${outfreq}%pressure=${pressure}%temperature=${temperature}%ifreconly=${ifreconly}%outputformat=${outputformat}%trajext=${trajext}%preveng=${preveng}%rstfrom=${rstfrom}%forcefield=${forcefield}%seqfile=${RSTREFDIR}${rstfrom}_SEQ.seq%cgenff=${CGENFFEXE}%neuiontype=${neuiontype}%solventmodel=${solventmodel}%outputgroup=${outputgroup}%ensemble=${ensemble}%lastNr=${lastNr}%parmfile=${parmfile}%wat_itp=${wat_itp}"| tr "%" "\n" > ${TMPDIR}tmp.txt;

        echo "Restarting previous CAMPARI simulation to Gromacs at $(date +%b.%d.%y-%H:%M)";
        insertInd=$(cat -n ${REFDIR}singleMD/exprst.sh | grep INSERT_VARIABLE_HERE | head -1 | awk '{print $1}');
        sed -i "${insertInd}r  ${TMPDIR}tmp.txt" ${REFDIR}singleMD/exprst.sh;
        chmod +x ${REFDIR}singleMD/exprst.sh;

        srun --wait=7200 -v  -n ${nrcopy} --cpus-per-task=${SLURM_CPUS_PER_TASK}  --ntasks-per-core=${SLURM_NTASKS_PER_CORE} --mpi=pmix_v2 ${REFDIR}singleMD/exprst.sh > ${TMPDIR}EXP_restarts.log 2>&1;
        checklast=$(ls -d  ${REFDIR}singleMD/[CTRL][0-9][0-9][0-9]${tmpfolder} | awk -F / '{print $(NF)}' | awk '{print substr($0,2,3)}' | sort -n | tail -1 );
        if [ ${lastNr} -eq ${checklast} ]; then  echo "Fatal: No new Trajectory is detected."; exit 62; fi

    fi
    last_id=$(ls -d  ${REFDIR}singleMD/[CTRL][0-9][0-9][0-9]${tmpfolder} | awk -F / '{print $(NF)}' | sort -nk1.2,1.4 | tail -1 );
    echo "In the past simulation, the latest generated trajectory ID is ${last_id}";
    echo "Explicit restart simulation ends at $(date +%b.%d.%y-%H:%M)";
    

    check1=$(egrep "Fatal:|exit code|The designated file.*is not found or too small" ${TMPDIR}EXP_restarts.log );
    Exit_code=$(egrep "Fatal:|exit code|The designated file.*is not found or too small" ${TMPDIR}EXP_restarts.log | sed "s|.*exit code ||g" | awk '{print $1}');
    if [ ${#check1} -ne 0 ]; then  Exit_code=${Exit_code:-54}; exit ${Exit_code}; fi

    check2=$(egrep "Simulation Success" ${TMPDIR}EXP_restarts.log | head -1 );
    if [ ${#check2} -ne 0 ]; then  echo "Simulation successed"; exit 1;
    else  echo "No explicit exit code – Fail in later stage"; exit 68;
    fi

elif ([[ ${ifrestart} == "true" ]] && [[ ${ifimp} == "true" ]]) ; then    # Restart an impolicit solvent simulation 
    RSTREFDIR=${REFDIR}singleMD/${rstfrom}/;
    if [[ ${forcefield} == "ff1" ]]; then forcefield=abs4.2_charmm36.prm;
    elif [[ ${forcefield} == "ff2" ]]; then forcefield=abs4.2_opls.prm;
    else forcefield=abs4.2_charmm36.prm;
    fi
    if [[ ${outputformat} == "netcdf" ]]; then  outputformat=5; trajext=nc;
    elif [[ ${outputformat} == "xtc" ]]; then outputformat=4; trajext=xtc;
    elif [[ ${outputformat} == "dcd" ]]; then outputformat=3; trajext=dcd;
    elif [[ ${outputformat} == "pdb" ]]; then outputformat=2; trajext=pdb;
    fi

    pdbcenter=$(egrep "ATOM|HETATM" ${RSTREFDIR}${rstfrom}_PDB.pdb | awk -v x=0 -v y=0 -v z=0 -v kk=0 '{if (substr($0,1,4) == "ATOM") {x=x+1.0*substr($0,31,8); y=y+1.0*substr($0,39,8); z=z+1.0*substr($0,47,8); kk=kk+1}; if (kk > 0) {print x/kk,y/kk,z/kk}}' | tail -n1);
    echo system length is ${systemlength}
    # need to modularize shape can change
    systemvol=$(echo "${systemlength}" | awk '{print 4.18879*$1*$1*$1}');

    # Step1: extract the tar file 
    if [ -s ${RSTREFDIR}${rstfrom}_RSTPKG.tar ]; then
        rm -f ${TMPDIR}checkpoint.cpt ${TMPDIR}checkpoint.gro ${TMPDIR}checkpoint.pdb ${TMPDIR}checkpoint.mol2 ${TMPDIR}checkpoint.seq 
        rm -f ${TMPDIR}rec_lig_index.ndx ${TMPDIR}topol.top ${TMPDIR}MDL.str ${TMPDIR}mdl.prm ${TMPDIR}mdl.itp ${TMPDIR}posre.itp;
        rm -f ${TMPDIR}autoPatch_*
        tar -xvf ${RSTREFDIR}${rstfrom}_RSTPKG.tar -C ${TMPDIR};
        fileCheck ${TMPDIR}checkpoint.pdb  "Missing restart files." 63;  fileCheck ${TMPDIR}checkpoint.seq  "Missing restart files." 63; fileCheck ${TMPDIR}checkpoint.mol2 "Missing restart files."  63;
    else
        echo "No revious restart package is found " 1>&2; exit 10;
    fi


    echo -e ${imp_sim_key} | tr '%' '\n' | sed -e "s|scriptdir|${SCRIPTDIR}|g" -e "s|campadatadir|${CAMPDATADIR}|g" -e "s|campaparamsdir|${CAMPPARM}|g" -e "s|tmpfolder|${REFDIR}|g" \
    -e "s|THESEQFILE|${TMPDIR}checkpoint.seq|g" -e "s|THEPDBFILE|${TMPDIR}checkpoint.pdb|g" -e "s|THEMOL2FILE|${TMPDIR}checkpoint.mol2|g" \
    -e "s|THEBASENAME|SMDIMP|g" -e "s|THESIZE|${systemlength}|g" -e "s|THEORIGIN|${pdbcenter}|g" -e "s|THESHAPE|2|g" -e "s|THEBOUND|4|g" \
    -e "s|THENRSTEP|${nrsteps}|g" -e "s|THEEQUIL|0|g" -e "s|THEDYNAMICS|${simmode}|g" -e "s|THE_TEMPERATURE|${temperature}|g" -e "s|THE_TIMESTEP|${timestep}|g" \
    -e "s|THEMCFIRST|${inimcsteps}|g" -e "s|THE_MC_CYCLE|${hybmcsteps}|g" -e "s|THE_MD_CYCLE|${hybmdsteps}|g" \
    -e "s|THE_ENSEMBLE|${ensemble}|g" -e "s|THETRAJFORMAT|${outputformat}|g" -e "s|THEOUTFREQ|${outfreq}|g" \
    -e "s|THEIFLIGAND|1|g" -e "s|PARMFILE|${forcefield}|g" > ${REFDIR}singleMD/IMP_rst.key;

    if [[ ${ifreconly} == "true" ]]; then sed -i -e "s|MOL2SCREEN.*|MOL2SCREEN 0|g" ${REFDIR}singleMD/IMP_rst.key ; fi
    if (( $(echo "${prodsctor} > 0"|bc) )); then torfile=${REFDIR}singleMD/TorFile.txt; PrepareTorfile ${torfile} 20 ${REFDIR}singleMD/rec.pdb; sed -i -e "s|TORFILE .*|TORFILE ${torfile}|g" -e "s|SC_TOR .*|SC_TOR ${prodsctor}|g" ${REFDIR}singleMD/IMP_rst.key; fi

    addPatches ${TMPDIR} ${REFDIR}singleMD/IMP_rst.key; 

    # prepare script file and parameters need by this script. 
    cat ${SCRIPTDIR}smd_impprod.sh > ${REFDIR}singleMD/IMP_Restart.sh;
    echo "TMPDIR=${TMPDIR}%REFDIR=${REFDIR}%SCRIPTDIR=${SCRIPTDIR}%CAMPDATADIR=${CAMPDATADIR}%CAMPPARM=${CAMPPARM}%ifreconly=\"${ifreconly}\"%trajext=${trajext}%seqfile=${TMPDIR}checkpoint.seq%CAMPEXE_NOMPI=${CAMPEXE_NOMPI}%CAMPMPI_MPI_THREADEXE=${CAMPMPI_MPI_THREADEXE}%nrsteps=${nrsteps}%timestep=${timestep}%outputformat=${outputformat}%forcefield=${forcefield}%neuiontype=${neuiontype}%outputgroup=${outputgroup}%systemlength=${systemlength}%keyfile=${REFDIR}singleMD/IMP_rst.key%lastNr=${lastNr}%parmfile=${parmfile}%slotsel=${slotsel}%outfreq=${outfreq}" | tr '%' '\n' > ${TMPDIR}tmp.txt; 
    echo "Implicit restart simulation starts at $(date +%b.%d.%y-%H:%M)";
    insertInd=$(cat -n ${REFDIR}singleMD/IMP_Restart.sh | grep INSERT_VARIABLE_HERE | awk '{print $1}');
    sed -i "${insertInd}r  ${TMPDIR}tmp.txt" ${REFDIR}singleMD/IMP_Restart.sh;
    chmod +x ${REFDIR}singleMD/IMP_Restart.sh;

    srun --wait=7200 -v  -n ${nrcopy} --cpus-per-task=${SLURM_CPUS_PER_TASK}  --ntasks-per-core=${SLURM_NTASKS_PER_CORE} --mpi=pmix_v2 ${REFDIR}singleMD/IMP_Restart.sh > ${TMPDIR}IMP_restarts.log 2>&1;
    checklast=$(ls -d  ${REFDIR}singleMD/[CTRL][0-9][0-9][0-9]${tmpfolder} | awk -F / '{print $(NF)}' | awk '{print substr($0,2,3)}' | sort -n | tail -1 );
    if [ ${lastNr} -eq ${checklast} ]; then  echo "Fatal: No new Trajectory is detected."; exit 62; fi
    last_id=$(ls -d  ${REFDIR}singleMD/[CTRL][0-9][0-9][0-9]${tmpfolder} | awk -F / '{print $(NF)}' | sort -nk1.2,1.4 | tail -1 );
    echo "In the past simulation, the latest generated trajectory ID is ${last_id}";
    echo "Implicit restart simulation ends at $(date +%b.%d.%y-%H:%M)";

    check1=$(egrep "Fatal:|exit code|The designated file.*is not found or too small" ${TMPDIR}IMP_restarts.log);
    Exit_code=$(egrep "Fatal:|exit code|The designated file.*is not found or too small" ${TMPDIR}IMP_restarts.log | sed "s|.*exit code ||g" | awk '{print $1}');
    if [ ${#check1} -ne 0 ]; then  Exit_code=${Exit_code:-54}; exit ${Exit_code}; fi

    check2=$(egrep "Simulation Success" ${TMPDIR}IMP_restarts.log | head -1);
    if [ ${#check2} -ne 0 ]; then  echo "Simulation successed"; exit 1;
    else  echo "No explicit exit code – Fail in later stage"; exit 68;
    fi

fi

