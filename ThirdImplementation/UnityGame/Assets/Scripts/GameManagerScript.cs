using System;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;

public class GameManagerScript : MonoBehaviour
{
    public GameObject player;
    public static Action lockdownInitiated;
    public static Action alarmInitiated;
    public static GameManagerScript instance;
    public bool hasHammer;
    public bool leverPressed;
    public bool diamondStolen;

    public bool alarmRaised;
    public bool inLockdown;
    public TextMeshProUGUI winText;
    public TextMeshProUGUI loseText;
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        hasHammer = false;
        instance = this;
        diamondStolen = false;
        leverPressed = false;
        alarmRaised = false;
        inLockdown = false;
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public async void RaiseAlarm()
    {
        if (!alarmRaised)
        {
            alarmRaised = true;
            alarmInitiated.Invoke();
            await Task.Delay(23700);
            lockdownInitiated.Invoke();
            inLockdown = true;
        }
    }
    public void WinGame()
    {
        Destroy(player.gameObject);
        winText.gameObject.SetActive(true);
        
    }

    public void LoseGame()
    {
        Destroy(player.gameObject);
        loseText.gameObject.SetActive(true);
        
    }
}
