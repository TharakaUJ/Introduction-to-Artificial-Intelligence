(define (domain fetch-domain)
  (:requirements :strips :typing)
  (:types room treat)
  
  (:predicates
    (at ?r - room)           ; the agent is in room ?r
    (connected ?r1 ?r2 - room) ; room ?r1 is connected to room ?r2
    (treat-at ?t - treat ?r - room) ; treat ?t is at room ?r
    (has-treat ?t - treat)          ; agent has treat ?t
  )

  (:action pickup
    :parameters (?t - treat ?r - room)
    :precondition (and (at ?r) (treat-at ?t ?r))
    :effect (and
              (not (treat-at ?t ?r))
              (has-treat ?t)
            )
  )

  (:action drop
    :parameters (?t - treat ?r - room)
    :precondition (and (at ?r) (has-treat ?t))
    :effect (and
              (not (has-treat ?t))
              (treat-at ?t ?r)
            )
  )
  
  (:action move
    :parameters (?from ?to - room)
    :precondition (and (at ?from) (connected ?from ?to))
    :effect (and
              (not (at ?from))
              (at ?to)
            )
  )
)

