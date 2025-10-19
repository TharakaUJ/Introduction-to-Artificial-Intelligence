(define (problem fetch-treat)
    (:domain fetch-domain)
    (:objects
        roomA roomB roomC roomD - room
        treat1 - treat
    )
    (:init
        (at roomA)
        (treat-at treat1 roomD)
        (connected roomA roomB)
        (connected roomB roomC)
        (connected roomC roomD)
        (connected roomD roomC)
        (connected roomC roomB)
        (connected roomB roomA)
    )
    (:goal
        (and
            (at roomA)
            (has-treat treat1)
        )
    )
)