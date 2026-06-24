# TODO: rework?
# visualization.py
#   code for timing plots should go here?



def save_timing_plot(timing_history, fig, ax, plot_name):
    avg_predict_time = np.mean(timing_history["predict_action"])
    std_predict_time = np.std(timing_history["predict_action"])
    logging.info(f"Average predict_action time: {avg_predict_time:.1f} ms")
    logging.info(f"Std dev predict_action time: {std_predict_time:.1f} ms")

    ax.clear()
    iterations = range(len(timing_history["camera_capture"]))
    ax.plot(iterations, timing_history["camera_capture"], label="camera_capture")
    ax.plot(iterations, timing_history["obs_processing"],  label="obs_processing")
    ax.plot(iterations, timing_history["predict_action"],  label="predict_action")
    ax.legend(loc="upper left")
    ax.set_xlabel("iteration")
    ax.set_ylabel("ms")
    ax.set_title("per-iteration timing")

    stats_text = f"predict_action avg: {avg_predict_time:.1f} ms\nstd dev: {std_predict_time:.1f} ms"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), fontsize=10)

    plt.ioff()
    plt.savefig(f"inference_timing_plots/{plot_name}.png")
    plt.close()



def update_live_plot(timing_history, ax):
    ax.clear()
    iterations = range(len(timing_history["camera_capture"]))
    ax.plot(iterations, timing_history["camera_capture"], label="camera_capture")
    ax.plot(iterations, timing_history["obs_processing"],  label="obs_processing")
    ax.plot(iterations, timing_history["predict_action"],  label="predict_action")
    ax.legend()
    ax.set_xlabel("iteration")
    ax.set_ylabel("ms")
    ax.set_title("per-iteration timing")
    plt.pause(0.001)




def save_attention_video(viz_frames, fps):
    if not viz_frames:
        print("did not make video, viz_frames DNE")
        return
    video_path = f"attention_videos/attn_vis_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    imageio.mimwrite(video_path, viz_frames, fps=fps, codec="libx264")
    logging.info(f"Saved attention visualization to {video_path}")