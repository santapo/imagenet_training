from pytorch_lightning.utilities.rank_zero import rank_zero_only


AIVN_LOGO = """\033[92m
              :!77!~.                    .~77~. 
            :YGGGGGGP7                  ^PGGGGP~
           :5GGGGGGGGG?                 JGGGGGGJ
          .5GGGGGGGGGGG7                JGGGGGGJ
          JGGGGG5JGGGGGG~               JGGGGGGJ
         ?GGGGGG^ YGGGGGP^              JGGGGGGJ
        !GGGGGG7  :PGGGGGP:             JGGGGGGJ
       ~PGGGGGJ    ^GGGGGG5.            JGGGGGGJ
      :PGGGGG5.     7GGGGGGY            JGGGGGGJ
     .5GGGGGP^       JGGGGGG?           JGGGGGGJ
     JGGGGGG!        .YGGGGGG7          JGGGGGGJ
    ?GGGGGGPJJJJJJJJJ?YGGGGGGG!         JGGGGGGJ
   !GGGGGGGGGGGGGGGGGGGGGGGGGGP^        JGGGGGGJ
  ~GGGGGGP55555555555555PGGGGGGP:       JGGGGGGJ
 :PGGGGGG!...............YGGGGGG5.      JGGGGGGJ
.5GGGGGG?                :5GGGGGGY      JGGGGGGJ
7GGGGGGY                  ^PGGGGGG~     JGGGGGGJ
^5GGGGY.                   ^5GGGGY:     ^5GGGG5^
 .^!!^                      .^!~^        .^!!^. 
\033[0m"""

@rank_zero_only
def print_logo():
    print(AIVN_LOGO)