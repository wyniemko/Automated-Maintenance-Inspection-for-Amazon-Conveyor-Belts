[Unit]
Description=My Flask App
After=network.target

[Service]
User=myuser
WorkingDirectory=/path/to/app
Environment="PATH=/home/myuser/myenv/bin"
ExecStart=/home/myuser/myenv/bin/python /path/to/app/run.py
Restart=always

[Install]
WantedBy=multi-user.target


-- reload with: sudo systemctl daemon-reload

-- start with: sudo systemctl start myapp

-- check status: sudo systemctl status myapp


# Open crontab
crontab -e

# Add the following line to the end of the file:
0 10 * * * cd /path/to/your/project && /path/to/python /path/to/your/project/belt_monitor.py -s 12 -l 15 && /usr/bin/firefox /path/to/your/project/results.html

