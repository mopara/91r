alias clr='clear && printf "\e[3J"'
alias src='echo "reloading /home/91r.sh"; source "/home/91r.sh"'

alias ls='ls --color=auto -a'
alias ll='ls --lh'
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

function mlrun {
  pushd /home/ra_login/91r
  logfile=src/log/"$(date '+20%y-%m-%d-%H-%M-%S')-$1.txt"
  cat src/"$1".py > "$logfile"
  echo "[EOF]" >> "$logfile"
  python -u src/"$1".py -s -b "$2" -n "$3" -i mnist/train-images-idx3-ubyte."$4" -j mnist/t10k-images-idx3-ubyte."$4" | tee -a "$logfile"
  popd
}

function keras_ae {
  mlrun keras_ae 256 "$1" npy
}

function 91r {
  mlrun 91r 128 "$1" T
}
