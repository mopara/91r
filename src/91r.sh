alias clr='clear && printf "\e[3J"'
alias src='echo "reloading /home/vae.sh"; source "/home/vae.sh"'

alias v2='source activate vae'
alias d='source deactivate'

bind '"\t":menu-complete'
bind 'set completion-ignore-case on'

function pull {
  pushd '/home/ra_login/91r'
  git pull origin master
  popd
}
