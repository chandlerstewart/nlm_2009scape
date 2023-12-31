package core.game.node.entity.player.link.audio;

import core.game.node.entity.player.Player;
import core.game.world.map.Location;
import core.game.world.map.MapDistance;
import core.game.world.map.RegionManager;
import core.net.packet.PacketRepository;
import core.net.packet.context.DefaultContext;
import core.net.packet.out.AudioPacket;

import java.util.List;

/**
 * Manages audio for a player.
 * @author Vexia
 */
public class AudioManager {

	/**
	 * The player instance.
	 */
	private final Player player;

	/**
	 * Constructs a new {@Code AudioManager} {@Code Object}
	 * @param player the player.
	 */
	public AudioManager(Player player) {
		this.player = player;
	}

	/**
	 * Sends an audio packet.
	 * @param audioId the audio id.
	 */
	public void send(int audioId) {
		send(audioId, false);
	}

	/**
	 * Sends an audio.
	 * @param audioId the audio id.
	 * @param global the global.
	 */
	public void send(int audioId, boolean global) {
		send(new Audio(audioId), global);
	}

	/**
	 * Sends an audio packet.
	 * @param audioId the audio id.
	 * @param volume the volume.
	 */
	public void send(int audioId, int volume) {
		send(new Audio(audioId, volume), false);
	}

	/**
	 * Sends an audio packet.
	 * @param audioId the audio id.
	 * @param volume the volume.
	 * @param delay the delay.
	 */
	public void send(int audioId, int volume, int delay) {
		send(new Audio(audioId, volume, delay), false);
	}

	/**
	 * Sends an audio packet.
	 * @param audioId the audio id.
	 * @param volume the volume.
	 * @param delay the delay.
	 * @param radius the distance the sound can be heard.
	 * @param loc the location.
	 */
	public void send(int audioId, int volume, int delay, int radius, Location loc) {
		send(new Audio(audioId, volume, delay, radius), false, loc);
	}

	/**
	 * Sends an audio packet.
	 * @param audio the audio.
	 */
	public void send(Audio audio) {
		send(audio, false);
	}

    public void send(Audio audio, boolean global) {
        send(audio, global, null);
    }

	/**
	 * Sends an audio packet.
	 * @param audio the audio.
	 * @param global if globally heard.
	 */
	public void send(Audio audio, boolean global, Location loc) {
		if (global) {
			send(audio, RegionManager.getLocalPlayers(player, MapDistance.SOUND.getDistance()), loc);
			return;
		}
		PacketRepository.send(AudioPacket.class, new DefaultContext(player, audio, loc));
	}

	/**
	 * Sends an audio packet for a bunch of players.
	 * @param audio the audio.
	 * @param players the players.
	 */
	public void send(Audio audio, List<Player> players, Location loc) {
		for (Player p : players) {
			if (p == null) {
				continue;
			}
			p.getAudioManager().send(audio, false, loc);
		}
	}

	/**
	 * Gets the player.
	 * @return the player
	 */
	public Player getPlayer() {
		return player;
	}

}
