:- use_module(library(scasp)).
:- use_module(library(http/json)).
:- consult('rules_temp.pl').

true_length([], 0).
true_length([_|T], N) :- true_length(T, N0), N is N0 + 1.

num_suspicious_things(Count) :-
    findall(X, noise(X), Noises),
    true_length(Noises, NoiseCount),

    findall(Y, suspicious_sighting(Y), Sightings),
    true_length(Sightings, SightCount),

    findall(Z, broken_vase(Z), Vases),
    true_length(Vases, VaseCount),

    Count is NoiseCount + SightCount + VaseCount.

num_broken_vases_by_player(Count) :-
    findall(_, broken_vase(player), BrokenByPlayer),
    true_length(BrokenByPlayer, Count).

is_valid(raise_alarm) :- not alarm_raised, num_suspicious_things(Count), Count > 2.

is_valid(raise_alarm) :- diamond_saw_broken, not alarm_raised.

player_acting_sussy :- player_in_restricted_area.

player_acting_sussy :- alarm_raised.

player_acting_sussy :- (num_broken_vases_by_player(Count), Count > 1).

is_valid(find_player_last) :- player_seen, player_acting_sussy.
is_valid(investigate_noise) :- noise(_).

is_valid(wander_randomly).

chosen_action(Action) :- is_valid(Action).

%! term_to_json(+Term, -Json) is det.
%  Recursively converts a Prolog term into a JSON-safe structure.
term_to_json(Term, Json) :-
    ( is_list(Term)
    -> maplist(term_to_json, Term, Json)
    ;  compound(Term)
    -> Term =.. [F|Args],
       maplist(term_to_json, Args, ArgsJson),
       Json = _{functor: F, args: ArgsJson}
    ;  Json = Term
    ).

bindings_to_json([], _{}).
bindings_to_json([Name=Value|T], Json) :-
    term_to_json(Value, ValueJson),
    bindings_to_json(T, RestJson),
    Json = RestJson.put(Name, ValueJson).

main :-
    current_prolog_flag(argv, [QueryAtom]),
    catch(
        (
            term_string(Query, QueryAtom, [variable_names(Bindings)]),

            findall(
                _{
                    bindings: BindingsJson,
                    model: ModelJson
                },
                (
                    scasp(Query, [model(Model)]),
                    bindings_to_json(Bindings, BindingsJson),
                    term_to_json(Model, ModelJson)
                ),
                Solutions
            ),

            (   Solutions \= []
            ->  Result = _{
                    ok: true,
                    solutions: Solutions
                }
            ;   Result = _{
                    ok: false,
                    error: "no solution"
                }
            )
        ),
        Error,
        (
            term_string(Error, ErrStr),
            Result = _{
                ok: false,
                error: ErrStr
            }
        )
    ),
    json_write_dict(current_output, Result),
    nl.

:- initialization(main, main).