#!/bin/bash -l 

CAMPEXE=${CAMPARI_EXE:-/software/campari/Gnu/bin/campari}
trajdir="/Weiss/PDBBind_trajs/C001C4001O5A"


########################################################################################
# trajtop="${trajdir}/checkpoint.pdb"
# Prepare the mapped pdb file for the trajectory conversion
FFTYPE="CHARMM"
trajseq="${trajdir}/checkpoint.seq"
systemsize=$(tail -1 "${trajdir}/checkpoint.gro" | awk '{print $1*10, $2*10, $3*10}')
CAMPARI_KEY_TRAJCONV="""FMCSC_BOUNDARY 1
FMCSC_SHAPE 1
FMCSC_ENSEMBLE 1
FMCSC_UAMODEL 0
FMCSC_ORIGIN 0 0 0
FMCSC_SIZE @THESIZE@
FMCSC_PDBANALYZE 1
FMCSC_UNSAFE 1
FMCSC_SEQFILE @TEMPLATE_SEQ@
FMCSC_SEQREPORT 1
FMCSC_PDB_TEMPLATE @TEMPLATE_PDB@
PARAMETERS @CAMP_DIR@/params/abs4.2_charmm36.prm
FMCSC_SYBYLLJMAP @CAMP_DIR@/params/abs4.2.ljmap
FMCSC_BASENAME ALIGN
FMCSC_DYNAMICS 2
FMCSC_NRSTEPS 9999999
FMCSC_DISABLE_ANALYSIS 1
FMCSC_SC_IPP 0.0
FMCSC_N2LOOP 0
FMCSC_CUTOFFMODE 4
FMCSC_PDB_FORMAT 3    # Input trajectory format
FMCSC_XTCFILE @TRAJFILE@
FMCSC_XYZPDB 5        # Output trajectory format 
FMCSC_XYZ_FORCEBOX 1
FMCSC_XYZ_SOLVENT 1
FMCSC_PDB_AUXINFO 3
FMCSC_NRTHREADS 4
FMCSC_FLUSHTIME 0.5
FMCSC_XYZOUT 1
FMCSC_PDB_R_CONV 3
FMCSC_PDB_W_CONV 3
FMCSC_ALIGNCALC 1
FMCSC_ALIGNFILE @ALIGNFILE@
"""


c=0
ls -d ${trajdir}/job* | while read dir; do 
  trajfile=${dir}/rst.xtc
  gmx editconf -f ${dir}/rst.gro -o /tmp/out.pdb
  # Get the atomic index of CA atoms for alignment
  grep "CA" /tmp/out.pdb | awk '{print substr($0,23,4)}' | awk '{print $1}' > /tmp/ca.idx

  # Correct the auxiliary information in the PDB file
  awk -v ff=${FFTYPE} -v fn=${trajseq} 'BEGIN{crsi="X"; crsn="XXXXX";} {if ((substr($0,1,6) == "ATOM  ") || (substr($0,1,6) == "HETATM")) {
  atn=substr($0,13,4); rsi=1*substr($0,23,4);
  if (rsi != crsi) {crsi = rsi; getline trsn<fn; rsn=substr(trsn,1,3); if (length(rsn) == 1) {rsn = rsn "  "} else if (length(rsn) == 2) {rsn =  rsn " "}};
  bc = substr(rsn,3,1);
  if (((substr(rsn,1,2) == "RI") || (substr(rsn,1,2) == "RP")) && ((bc == "A") || (bc == "G") || (bc == "U") || (bc == "T") || (bc == "C"))) {
    if (atn == "H2'\'\''") {atn = " H2*"} else if (((atn == " H2'\''") && (ff != "AMBER")) || ((ff == "GROMOS") && (atn == " H2*"))) {atn = " HO2"};
  };
  if (((substr(rsn,1,2) == "RI") || (substr(rsn,1,2) == "RP") || (substr(rsn,1,2) == "DP") || ((substr(rsn,1,2) == "DI"))) && ((bc == "A") || (bc == "G") || (bc == "U") || (bc == "T") || (bc == "C"))) {
  if (atn == " H3T") {atn = " HO3"} else if ((atn == " H5T") && ((substr(rsn,1,2) == "RI") || (substr(rsn,1,2) == "DI"))) {atn = " HO5"} else if (atn == " H5T") {atn = " HOP"};
  };
  if ((atn == " HAF") && (rsn == "FOR")) {atn = " H  ";}
  if ((atn == " C  ") && (rsn == "NME")) {atn = " CH3";}
  if ((ff == "AMBER") && (substr(trsn,4,2) == "_C")) {
    if (atn == " OXT") {atn = "2OXT"} else if (atn == " O  ") {atn = "1OXT"};
  }
  if (((substr(rsn,1,2) == "RP") || (substr(rsn,1,2) == "DP")) && ((bc == "A") || (bc == "G") || (bc == "U") || (bc == "T") || (bc == "C"))) {
  if (atn == " O5T") {atn = "2O3*"};
  if ((atn == " OP3") && (ff == "AMBER")) {atn = "2O3*"};
  if ((ff == "GROMOS") && (substr(trsn,4,2) == "_N")) {rsn = substr(rsn,1,1) "X" bc;}
  else if ((ff == "GROMOS") && (substr(trsn,4,2) == "_C")) {rsn = substr(rsn,1,1) "Y" bc;}
  } else if ((ff == "GROMOS") && (rsn == "EOH")) {if (atn == " EC1") {atn = " CB "} else if (atn == " EC2") {atn = " CT "} else if (atn == " EH ") {atn = " HO "} else if (atn == " EO ") {atn = " O  "}
  } else if ((ff == "OPLS") && (rsn == "EOH")) {if (atn == " CB ") {atn = " CT "} else if (atn == " CA ") {atn = " CB "} else if (substr(atn,2,2) == "HB") {atn = " HT" substr(atn,4,1)} else if (substr(atn,2,2) == "HA") {atn = " HB" substr(atn,4,1)}
  };
  prsn = substr($0,18,4); print substr($0,1,12) atn " " rsn " " substr($0,22,100)} else {print}}' /tmp/out.pdb > /tmp/out.pdb.tmp && mv /tmp/out.pdb.tmp /tmp/out.pdb

  
  echo "${CAMPARI_KEY_TRAJCONV}" | sed -e "s|@THESIZE@|${systemsize}|g" \
  -e "s|@TEMPLATE_SEQ@|${trajseq}|g" -e "s|@TEMPLATE_PDB@|/tmp/out.pdb|g" \
  -e "s|@CAMP_DIR@|${CAMPARI_DIR}|g" -e "s|@TRAJFILE@|${trajfile}|g" \
  -e "s|@ALIGNFILE@|/tmp/ca.idx|g"> /tmp/campari_${c}.key
  cd /tmp
  ${CAMPEXE} -k campari_${c}.key

  let c=c+1
  break
done 


# Chimera command
# open ALIGN_traj.nc start 1 step 50 ; select protein|:LIG ; save test.pdb format pdb selectedOnly true allCoordsets true

