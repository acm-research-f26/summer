:- use_module(library(scasp)).
:- use_module(library(http/json)).
:- consult('rules.pl').

count_list([], 0).
count_list([_|T], N) :- count_list(T, N0), N is N0 + 1.

num_suspicious_things(Count) :-
    findall(X, noise(X), Noises),
    count_list(Noises, NoiseCount),
    findall(Y, suspicious_sighting(Y), Sightings),
    count_list(Sightings, SightCount),
    findall(Z, broken_vase(Z), Vases),
    count_list(Vases, VaseCount),
    Count is NoiseCount + SightCount + VaseCount.

num_broken_vases_by_player(Count) :-
    findall(_, broken_vase(player), BrokenByPlayer),
    count_list(BrokenByPlayer, Count).

is_valid(raise_alarm) :- not alarm_raised, diamond_saw_broken, num_suspicious_things(Count), Count > 2.
player_acting_sussy :- player_in_restricted_area.
player_acting_sussy :- alarm_raised.
player_acting_sussy :- (num_broken_vases_by_player(Count), Count > 1).
is_valid(find_player_last) :- player_seen, player_acting_sussy.
is_valid(investigate_noise) :- noise(_).
is_valid(wander_randomly).