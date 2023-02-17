#!/bin/bash -l 

# Usage: bash ${0} <mol2 file>
# With xxxxxx.mol2 as input:
# It automatically outputs the xxxxxx_clean.mol2 and xxxxxx.sdf;

# This script is primarily used for cleaning the lone-pair atoms in the ACGui docked poses.

mol2file=${1};

if [ ${#mol2file} -eq 0 ]; then 
  echo "Fatal: Please define a mol2 file to clean" 1>&2;
  echo "Usage: " 1>&2; 
  echo "  \$bash ${0} <mol2 file>" 1>&2; 
  exit 0
elif ! [ -f ${mol2file} ]; then
	echo "Fatal: File ${mol2file} is not a valid file." 1>&2;
  echo "Usage: " 1>&2;
  echo "  \$bash ${0} <mol2 file>" 1>&2;
  exit 0
fi

mol2out=$(echo ${mol2file} | sed 's|.mol2||g')_clean.mol2;
sdfout=$(echo ${mol2file} | sed 's|.mol2||g').sdf;

[ -f ${mol2file} ] && touch ${mol2file};

python3 -c "from Doana import utils; utils.cleanMOL2LP('${mol2file}', '${mol2out}')";
obabel ${mol2out} -O ${sdfout};
