for game in Seaquest # SpaceInvaders Enduro BeamRider Breakout Qbert Seaquest Alien Pong
do
    for run in 4 5 6
    do
        msub -v EPISODES_PER_CPU=2,GAME=$game,CONFIG="configurations/sample_configuration.json",RUN=$run -l nodes=20:ppn=20,walltime=10:00:00 submit.sh
    done
done