alias clr='clear && printf "\e[3J"'
alias src='echo "reloading /home/91r.sh"; source "/home/91r.sh"'

alias ls='ls --color=auto -a'
alias v2='source activate 91r'
alias d='source deactivate'

bind '"\t":menu-complete'
bind 'set completion-ignore-case on'

function pull {
  pushd /home/ra_login/91r
  git pull origin master
  source /home/ra_login/91r/src/91r.sh
  popd
}

function keras_ae {
  pushd /home/ra_login/91r
  python src/keras_ae.py -s -b256 -n100 -i mnist/train-images-idx3-ubyte.npy -j mnist/t10k-images-idx3-ubyte.npy | tee log/"$(date '+20%y-%m-%d-%H-%M-%S.txt')"
  popd
}

function 91r {
  pushd /home/ra_login/91r
  python src/91r.py -s -c -b128 -n100 -i mnist/train-images-idx3-ubyte.T -j mnist/t10k-images-idx3-ubyte.T | tee log/"$(date '+20%y-%m-%d-%H-%M-%S.txt')"
  popd
}
