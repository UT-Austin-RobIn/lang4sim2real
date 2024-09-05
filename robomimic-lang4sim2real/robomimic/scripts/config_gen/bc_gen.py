from robomimic.scripts.config_gen.helper import *

def make_generator_helper(args):
    generator = get_generator(
        algo_name="bc",
        config_file=os.path.join(base_path, 'robomimic/exps/il/bc/base.json'),
        args=args,
    )

    generator.add_param(
        key="algo.gmm.enabled",
        name="gmm",
        group=0,
        values_and_names=[
            (True, "T"),
            # (False, "F"),
        ],
        hidename=True,
    )

    # basic env settings
    if args.env == 'calvin':
        generator.add_param(
            key="train.data",
            name="ds",
            group=-4,
            values=[
                os.path.join(base_path, path) for path in [
                    # 'datasets/calvin/turn_on_led_D_D.hdf5',
                    # 'datasets/calvin/turn_on_lightbulb_D_D.hdf5',
                    # 'datasets/calvin/open_drawer_D_D.hdf5',
                    # 'datasets/calvin/move_slider_left_D_D.hdf5',
                    # 'datasets/calvin/push_red_block_right_D_D.hdf5',
                    # 'datasets/calvin/stack_block_D_D.hdf5',
                    # 'datasets/calvin/place_in_slider_D_D.hdf5',

                    'datasets/calvin/slider_drawer_light_led/02_23_ld.hdf5',
                    # 'datasets/calvin/cleanup/02_23_ld.hdf5',
                    # 'datasets/calvin/cleanup_pink/02_25_ld.hdf5',
                ]
            ],
            value_names=[
                # "calv_turn_on_led_D_D",
                # "calv_turn_on_lightbulb_D_D",
                # "calv_open_drawer_D_D",
                # "calv_move_slider_left_D_D",
                # "calv_push_red_block_right_D_D",
                # "calv_stack_block_D_D",
                # "calv_place_in_slider_D_D",

                "calv_SDLiLe_D",
                # "calv_cleanup_D",
                # "calv_cleanup_pink_D",
            ],
            hidename=False,
        )

        generator.add_param(
            key="experiment.rollout.reset_to_dataset_start_states",
            name="resetds",
            group=-5,
            values_and_names=[
                (True, "T"),
                (False, "F"),
            ],
            hidename=False,
        )
        generator.add_param(
            key="experiment.rollout.horizon",
            name="",
            group=-4,
            values=[
                600,
                # 1000,
            ],
        )
        generator.add_param(
            key="train.hdf5_filter_key",
            name="ndemos",
            group=46,
            values_and_names=[
                (None, "full"),
                # ("30_demos", "30"),
                # ("100_demos", "100"),
                # ("50_demos", "50"),
                # ("25_demos", "25"),
            ],
            hidename=False,
        )
    elif args.env == 'kitchen':
        generator.add_param(
            key="train.data",
            name="",
            group=-4,
            values=[
                os.path.join(base_path, "datasets/d4rl/converted/kitchen_complete_v0.hdf5")
            ],
        )
    elif args.env == 'square':
        generator.add_param(
            key="train.data",
            name="",
            group=-4,
            values=[
                os.path.join(base_path, "datasets/square/ph/low_dim.hdf5")
            ],
        )
    elif args.env == 'transport':
        generator.add_param(
            key="train.data",
            name="",
            group=-4,
            values=[
                os.path.join(base_path, "datasets/transport/ph/low_dim.hdf5")
            ],
        )
    else:
        raise ValueError

    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "../expdata/{env}/{mod}/bc".format(
                env=args.env,
                mod=args.mod,
            )
        ],
    )
    generator.add_param(
        key="experiment.save.enabled",
        name="",
        group=-1,
        values=[
            False
        ],
    )

    return generator


if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)

