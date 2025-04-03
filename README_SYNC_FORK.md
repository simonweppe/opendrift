https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork

git fetch upstream
git checkout master
git merge upstream/master

commit then push 


---------------------------
to mount your code version to official docker 

# docker run -it  --rm -v /data2:/data2 -v /data:/data -v /data2/simon/github/opendrift:/code/  opendrift/opendrift:py3-v1.0.7
