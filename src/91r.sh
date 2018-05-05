alias clr='clear && printf "\e[3J"'
alias src='echo "reloading /home/91r.sh"; source "/home/91r.sh"'

alias ls='ls --color=auto -a'
alias v2='source activate 91r'
alias d='source deactivate'

bind '"\t":menu-complete'
bind 'set completion-ignore-case on'

function pull {
  pushd '/home/ra_login/91r'
  git pull origin master
  popd
}
